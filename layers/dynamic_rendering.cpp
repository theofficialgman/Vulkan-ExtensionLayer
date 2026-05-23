// Copyright (c) 2024 The Khronos Group Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// VK_KHR_dynamic_rendering emulation layer.
//
// Translates vkCmdBeginRenderingKHR / vkCmdEndRenderingKHR into traditional
// vkCmdBeginRenderPass / vkCmdEndRenderPass calls for drivers that do not
// natively support VK_KHR_dynamic_rendering.
//
// Also handles:
//   - vkCreateGraphicsPipelines with VkPipelineRenderingCreateInfo in pNext
//   - vkBeginCommandBuffer with VkCommandBufferInheritanceRenderingInfo in
//     the inheritance pNext (secondary command buffers)
//   - vkGetPhysicalDeviceFeatures2 / vkEnumerateDeviceExtensionProperties
//     to advertise the extension

#include <vulkan/vulkan_core.h>
#include <vulkan/vk_layer.h>

#include <cstring>
#include <mutex>
#include <unordered_map>
#include <string>
#include <algorithm>

// ============================================================================
// Export macro (matches the pattern used by other layers in this project)
// ============================================================================

#if defined(__GNUC__) || defined(__clang__)
#  define VEL_EXPORT __attribute__((visibility("default")))
#else
#  define VEL_EXPORT
#endif

// ============================================================================
// pNext chain helpers
// ============================================================================

static const void* find_pnext(const void* pNext, VkStructureType sType) {
    const VkBaseInStructure* p = static_cast<const VkBaseInStructure*>(pNext);
    while (p) {
        if (p->sType == sType) return p;
        p = p->pNext;
    }
    return nullptr;
}

// ============================================================================
// Chain info helpers (loader plumbing)
// ============================================================================

static VkLayerInstanceCreateInfo* get_instance_chain_info(
    const VkInstanceCreateInfo* pCreateInfo, VkLayerFunction func) {
    VkLayerInstanceCreateInfo* p =
        const_cast<VkLayerInstanceCreateInfo*>(
            static_cast<const VkLayerInstanceCreateInfo*>(pCreateInfo->pNext));
    while (p) {
        if (p->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO &&
            p->function == func)
            return p;
        p = const_cast<VkLayerInstanceCreateInfo*>(
            static_cast<const VkLayerInstanceCreateInfo*>(p->pNext));
    }
    return nullptr;
}

static VkLayerDeviceCreateInfo* get_device_chain_info(
    const VkDeviceCreateInfo* pCreateInfo, VkLayerFunction func) {
    VkLayerDeviceCreateInfo* p =
        const_cast<VkLayerDeviceCreateInfo*>(
            static_cast<const VkLayerDeviceCreateInfo*>(pCreateInfo->pNext));
    while (p) {
        if (p->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO &&
            p->function == func)
            return p;
        p = const_cast<VkLayerDeviceCreateInfo*>(
            static_cast<const VkLayerDeviceCreateInfo*>(p->pNext));
    }
    return nullptr;
}

// ============================================================================
// Cache key structures
// ============================================================================

// Key for render passes synthesized at vkCmdBeginRenderingKHR time.
// Encodes the full attachment state (formats, ops, layouts, samples).
struct RenderPassKey {
    uint32_t              color_count;
    VkFormat              color_formats[8];
    VkAttachmentLoadOp    color_load_ops[8];
    VkAttachmentStoreOp   color_store_ops[8];
    VkImageLayout         color_layouts[8];
    VkFormat              depth_format;     // VK_FORMAT_UNDEFINED = no depth
    VkAttachmentLoadOp    depth_load_op;
    VkAttachmentStoreOp   depth_store_op;
    VkImageLayout         depth_layout;
    VkAttachmentLoadOp    stencil_load_op;
    VkAttachmentStoreOp   stencil_store_op;
    VkImageLayout         stencil_layout;
    VkSampleCountFlagBits samples;
    uint32_t              view_mask;

    bool operator==(const RenderPassKey& o) const {
        return memcmp(this, &o, sizeof(*this)) == 0;
    }
};

// Key for pipeline-compatible render passes (vkCreateGraphicsPipelines with
// VkPipelineRenderingCreateInfo). Only format/samples/view_mask matter for
// pipeline compatibility; load/store ops do not.
struct PipelineRpKey {
    uint32_t              color_count;
    VkFormat              color_formats[8];
    VkFormat              depth_format;
    VkFormat              stencil_format;
    VkSampleCountFlagBits samples;
    uint32_t              view_mask;

    bool operator==(const PipelineRpKey& o) const {
        return memcmp(this, &o, sizeof(*this)) == 0;
    }
};

// Key for framebuffers. The render pass is part of the key because framebuffer
// compatibility is render-pass-specific.
struct FramebufferKey {
    VkRenderPass render_pass;
    uint32_t     attachment_count;
    VkImageView  attachments[9];  // up to 8 color + 1 depth/stencil
    uint32_t     width;
    uint32_t     height;
    uint32_t     layers;

    bool operator==(const FramebufferKey& o) const {
        return memcmp(this, &o, sizeof(*this)) == 0;
    }
};

// FNV-1a hash over the raw bytes of a key struct.
template<typename T>
struct ByteHash {
    size_t operator()(const T& v) const {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
        size_t h = 14695981039346656037ULL;
        for (size_t i = 0; i < sizeof(v); ++i) {
            h ^= p[i];
            h *= 1099511628211ULL;
        }
        return h;
    }
};

// ============================================================================
// Per-device and per-instance state
// ============================================================================

struct ImageViewInfo {
    VkFormat              format;
    VkSampleCountFlagBits samples;
};

struct DeviceData {
    VkDevice                     device;
    PFN_vkGetDeviceProcAddr      next_gdpa;

    // Forwarded device functions
    PFN_vkCreateRenderPass        CreateRenderPass;
    PFN_vkDestroyRenderPass       DestroyRenderPass;
    PFN_vkCreateFramebuffer       CreateFramebuffer;
    PFN_vkDestroyFramebuffer      DestroyFramebuffer;
    PFN_vkCmdBeginRenderPass      CmdBeginRenderPass;
    PFN_vkCmdEndRenderPass        CmdEndRenderPass;
    PFN_vkCreateGraphicsPipelines CreateGraphicsPipelines;
    PFN_vkCreateImage             CreateImage;
    PFN_vkDestroyImage            DestroyImage;
    PFN_vkCreateImageView         CreateImageView;
    PFN_vkDestroyImageView        DestroyImageView;
    PFN_vkBeginCommandBuffer      BeginCommandBuffer;

    // Image → sample count (populated at vkCreateImage)
    std::mutex image_mtx;
    std::unordered_map<VkImage, VkSampleCountFlagBits> image_map;

    // ImageView → {format, samples} (populated at vkCreateImageView)
    std::mutex iv_mtx;
    std::unordered_map<VkImageView, ImageViewInfo> iv_map;

    // Render pass caches
    std::mutex cache_mtx;
    std::unordered_map<RenderPassKey,  VkRenderPass, ByteHash<RenderPassKey>>  rendering_rp_cache;
    std::unordered_map<PipelineRpKey,  VkRenderPass, ByteHash<PipelineRpKey>>  pipeline_rp_cache;
    std::unordered_map<FramebufferKey, VkFramebuffer, ByteHash<FramebufferKey>> fb_cache;
};

struct InstanceData {
    VkInstance                              instance;
    PFN_vkGetInstanceProcAddr               next_gipa;
    PFN_vkGetPhysicalDeviceFeatures2        GetPhysicalDeviceFeatures2;
    PFN_vkEnumerateDeviceExtensionProperties EnumerateDeviceExtensionProperties;
};

// ============================================================================
// Global dispatch maps
// ============================================================================

static std::mutex g_dev_map_mtx;
static std::unordered_map<void*, DeviceData*> g_dev_map;

static std::mutex g_inst_map_mtx;
static std::unordered_map<void*, InstanceData*> g_inst_map;

// The Vulkan loader dispatch key is the first pointer-sized word of any
// dispatchable handle.
static void* dispatch_key(const void* handle) {
    return *reinterpret_cast<void* const*>(handle);
}

static DeviceData* get_dev(void* key) {
    std::lock_guard<std::mutex> lk(g_dev_map_mtx);
    auto it = g_dev_map.find(key);
    return it != g_dev_map.end() ? it->second : nullptr;
}

static InstanceData* get_inst(void* key) {
    std::lock_guard<std::mutex> lk(g_inst_map_mtx);
    auto it = g_inst_map.find(key);
    return it != g_inst_map.end() ? it->second : nullptr;
}

// ============================================================================
// Render pass creation helpers
// ============================================================================

// Build a VkRenderPass from a full rendering key (for vkCmdBeginRenderingKHR).
// initialLayout/finalLayout for each attachment are set to the imageLayout
// the app specified — it has already transitioned images to that layout before
// calling CmdBeginRendering. Setting both initial and final layout to the same
// value suppresses automatic layout transitions in the render pass.
static VkRenderPass create_rendering_rp(DeviceData* dev, const RenderPassKey& k) {
    VkAttachmentDescription descs[9] = {};
    VkAttachmentReference   color_refs[8] = {};
    VkAttachmentReference   depth_ref = {};

    uint32_t att = 0;
    uint32_t color_att_index[8];

    for (uint32_t i = 0; i < k.color_count && i < 8; ++i) {
        if (k.color_formats[i] == VK_FORMAT_UNDEFINED) {
            color_refs[i] = {VK_ATTACHMENT_UNUSED, VK_IMAGE_LAYOUT_UNDEFINED};
            color_att_index[i] = VK_ATTACHMENT_UNUSED;
            continue;
        }
        color_att_index[i] = att;
        VkAttachmentDescription& d = descs[att++];
        d.format         = k.color_formats[i];
        d.samples        = k.samples;
        d.loadOp         = k.color_load_ops[i];
        d.storeOp        = k.color_store_ops[i];
        d.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        d.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        d.initialLayout  = k.color_layouts[i];
        d.finalLayout    = k.color_layouts[i];
        color_refs[i]    = {color_att_index[i], k.color_layouts[i]};
    }

    bool has_ds = (k.depth_format != VK_FORMAT_UNDEFINED);
    uint32_t ds_att = VK_ATTACHMENT_UNUSED;
    if (has_ds) {
        ds_att = att;
        VkAttachmentDescription& d = descs[att++];
        d.format         = k.depth_format;
        d.samples        = k.samples;
        d.loadOp         = k.depth_load_op;
        d.storeOp        = k.depth_store_op;
        d.stencilLoadOp  = k.stencil_load_op;
        d.stencilStoreOp = k.stencil_store_op;
        d.initialLayout  = k.depth_layout;
        d.finalLayout    = k.depth_layout;
        depth_ref        = {ds_att, k.depth_layout};
    }

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = k.color_count;
    subpass.pColorAttachments       = k.color_count > 0 ? color_refs : nullptr;
    subpass.pDepthStencilAttachment = has_ds ? &depth_ref : nullptr;

    VkRenderPassCreateInfo rpci = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpci.attachmentCount = att;
    rpci.pAttachments    = att > 0 ? descs : nullptr;
    rpci.subpassCount    = 1;
    rpci.pSubpasses      = &subpass;

    VkRenderPassMultiviewCreateInfo mv = {VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO};
    uint32_t corr = k.view_mask;
    if (k.view_mask != 0) {
        mv.subpassCount         = 1;
        mv.pViewMasks           = &k.view_mask;
        mv.correlationMaskCount = 1;
        mv.pCorrelationMasks    = &corr;
        rpci.pNext              = &mv;
    }

    VkRenderPass rp = VK_NULL_HANDLE;
    dev->CreateRenderPass(dev->device, &rpci, nullptr, &rp);
    return rp;
}

// Build a pipeline-compatible VkRenderPass from a pipeline key.
// Uses LOAD/STORE ops and standard layouts — only format/samples/view_mask
// matter for pipeline compatibility per the Vulkan spec.
static VkRenderPass create_pipeline_rp(DeviceData* dev, const PipelineRpKey& k) {
    VkAttachmentDescription descs[9] = {};
    VkAttachmentReference   color_refs[8] = {};
    VkAttachmentReference   depth_ref = {};

    uint32_t att = 0;

    for (uint32_t i = 0; i < k.color_count && i < 8; ++i) {
        if (k.color_formats[i] == VK_FORMAT_UNDEFINED) {
            color_refs[i] = {VK_ATTACHMENT_UNUSED, VK_IMAGE_LAYOUT_UNDEFINED};
            continue;
        }
        VkAttachmentDescription& d = descs[att];
        d.format         = k.color_formats[i];
        d.samples        = k.samples;
        d.loadOp         = VK_ATTACHMENT_LOAD_OP_LOAD;
        d.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        d.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        d.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        d.initialLayout  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        d.finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_refs[i]    = {att, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        ++att;
    }

    // Combine depth and stencil into one attachment (the common case with
    // packed formats like D24_S8, D32_S8). If only one is requested, use
    // DONT_CARE for the other aspect's ops.
    bool has_depth   = (k.depth_format   != VK_FORMAT_UNDEFINED);
    bool has_stencil = (k.stencil_format != VK_FORMAT_UNDEFINED);
    VkFormat ds_fmt  = has_depth ? k.depth_format : k.stencil_format;
    bool has_ds      = (ds_fmt != VK_FORMAT_UNDEFINED);
    uint32_t ds_att  = VK_ATTACHMENT_UNUSED;

    if (has_ds) {
        ds_att = att;
        VkAttachmentDescription& d = descs[att++];
        d.format         = ds_fmt;
        d.samples        = k.samples;
        d.loadOp         = has_depth   ? VK_ATTACHMENT_LOAD_OP_LOAD        : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        d.storeOp        = has_depth   ? VK_ATTACHMENT_STORE_OP_STORE       : VK_ATTACHMENT_STORE_OP_DONT_CARE;
        d.stencilLoadOp  = has_stencil ? VK_ATTACHMENT_LOAD_OP_LOAD        : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        d.stencilStoreOp = has_stencil ? VK_ATTACHMENT_STORE_OP_STORE       : VK_ATTACHMENT_STORE_OP_DONT_CARE;
        d.initialLayout  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        d.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth_ref        = {ds_att, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    }

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = k.color_count;
    subpass.pColorAttachments       = k.color_count > 0 ? color_refs : nullptr;
    subpass.pDepthStencilAttachment = has_ds ? &depth_ref : nullptr;

    VkRenderPassCreateInfo rpci = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpci.attachmentCount = att;
    rpci.pAttachments    = att > 0 ? descs : nullptr;
    rpci.subpassCount    = 1;
    rpci.pSubpasses      = &subpass;

    VkRenderPassMultiviewCreateInfo mv = {VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO};
    uint32_t corr = k.view_mask;
    if (k.view_mask != 0) {
        mv.subpassCount         = 1;
        mv.pViewMasks           = &k.view_mask;
        mv.correlationMaskCount = 1;
        mv.pCorrelationMasks    = &corr;
        rpci.pNext              = &mv;
    }

    VkRenderPass rp = VK_NULL_HANDLE;
    dev->CreateRenderPass(dev->device, &rpci, nullptr, &rp);
    return rp;
}

static VkRenderPass get_or_create_rendering_rp(DeviceData* dev, const RenderPassKey& k) {
    std::lock_guard<std::mutex> lk(dev->cache_mtx);
    auto it = dev->rendering_rp_cache.find(k);
    if (it != dev->rendering_rp_cache.end()) return it->second;
    VkRenderPass rp = create_rendering_rp(dev, k);
    if (rp != VK_NULL_HANDLE) dev->rendering_rp_cache[k] = rp;
    return rp;
}

static VkRenderPass get_or_create_pipeline_rp(DeviceData* dev, const PipelineRpKey& k) {
    std::lock_guard<std::mutex> lk(dev->cache_mtx);
    auto it = dev->pipeline_rp_cache.find(k);
    if (it != dev->pipeline_rp_cache.end()) return it->second;
    VkRenderPass rp = create_pipeline_rp(dev, k);
    if (rp != VK_NULL_HANDLE) dev->pipeline_rp_cache[k] = rp;
    return rp;
}

static VkFramebuffer get_or_create_fb(DeviceData* dev, const FramebufferKey& k) {
    std::lock_guard<std::mutex> lk(dev->cache_mtx);
    auto it = dev->fb_cache.find(k);
    if (it != dev->fb_cache.end()) return it->second;

    VkFramebufferCreateInfo fbci = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbci.renderPass      = k.render_pass;
    fbci.attachmentCount = k.attachment_count;
    fbci.pAttachments    = k.attachments;
    fbci.width           = k.width;
    fbci.height          = k.height;
    fbci.layers          = k.layers;

    VkFramebuffer fb = VK_NULL_HANDLE;
    dev->CreateFramebuffer(dev->device, &fbci, nullptr, &fb);
    if (fb != VK_NULL_HANDLE) dev->fb_cache[k] = fb;
    return fb;
}

// ============================================================================
// Image / image-view tracking
// ============================================================================

static VKAPI_ATTR VkResult VKAPI_CALL layer_CreateImage(
    VkDevice device, const VkImageCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkImage* pImage)
{
    DeviceData* dev = get_dev(dispatch_key(device));
    VkResult r = dev->CreateImage(device, pCreateInfo, pAllocator, pImage);
    if (r == VK_SUCCESS) {
        std::lock_guard<std::mutex> lk(dev->image_mtx);
        dev->image_map[*pImage] = pCreateInfo->samples;
    }
    return r;
}

static VKAPI_ATTR void VKAPI_CALL layer_DestroyImage(
    VkDevice device, VkImage image, const VkAllocationCallbacks* pAllocator)
{
    DeviceData* dev = get_dev(dispatch_key(device));
    {
        std::lock_guard<std::mutex> lk(dev->image_mtx);
        dev->image_map.erase(image);
    }
    dev->DestroyImage(device, image, pAllocator);
}

static VKAPI_ATTR VkResult VKAPI_CALL layer_CreateImageView(
    VkDevice device, const VkImageViewCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkImageView* pView)
{
    DeviceData* dev = get_dev(dispatch_key(device));
    VkResult r = dev->CreateImageView(device, pCreateInfo, pAllocator, pView);
    if (r == VK_SUCCESS) {
        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
        {
            std::lock_guard<std::mutex> lk(dev->image_mtx);
            auto it = dev->image_map.find(pCreateInfo->image);
            if (it != dev->image_map.end()) samples = it->second;
        }
        std::lock_guard<std::mutex> lk(dev->iv_mtx);
        dev->iv_map[*pView] = {pCreateInfo->format, samples};
    }
    return r;
}

static VKAPI_ATTR void VKAPI_CALL layer_DestroyImageView(
    VkDevice device, VkImageView imageView,
    const VkAllocationCallbacks* pAllocator)
{
    DeviceData* dev = get_dev(dispatch_key(device));
    {
        std::lock_guard<std::mutex> lk(dev->iv_mtx);
        dev->iv_map.erase(imageView);
    }
    dev->DestroyImageView(device, imageView, pAllocator);
}

// ============================================================================
// CmdBeginRenderingKHR
// ============================================================================

static VKAPI_ATTR void VKAPI_CALL layer_CmdBeginRenderingKHR(
    VkCommandBuffer commandBuffer, const VkRenderingInfo* pRenderingInfo)
{
    DeviceData* dev = get_dev(dispatch_key(commandBuffer));
    if (!dev) return;

    const VkRenderingInfo& ri = *pRenderingInfo;

    // Determine the sample count from the first non-null image view.
    // All attachments must have the same sample count per spec.
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    {
        std::lock_guard<std::mutex> lk(dev->iv_mtx);
        for (uint32_t i = 0; i < ri.colorAttachmentCount && i < 8; ++i) {
            if (ri.pColorAttachments[i].imageView != VK_NULL_HANDLE) {
                auto it = dev->iv_map.find(ri.pColorAttachments[i].imageView);
                if (it != dev->iv_map.end()) { samples = it->second.samples; break; }
            }
        }
        if (samples == VK_SAMPLE_COUNT_1_BIT && ri.pDepthAttachment &&
            ri.pDepthAttachment->imageView != VK_NULL_HANDLE) {
            auto it = dev->iv_map.find(ri.pDepthAttachment->imageView);
            if (it != dev->iv_map.end()) samples = it->second.samples;
        }
    }

    // Build the render pass key.
    RenderPassKey rpk = {};
    rpk.samples     = samples;
    rpk.view_mask   = ri.viewMask;
    rpk.color_count = ri.colorAttachmentCount < 8 ? ri.colorAttachmentCount : 8;

    {
        std::lock_guard<std::mutex> lk(dev->iv_mtx);
        for (uint32_t i = 0; i < rpk.color_count; ++i) {
            const VkRenderingAttachmentInfo& ca = ri.pColorAttachments[i];
            if (ca.imageView == VK_NULL_HANDLE) {
                rpk.color_formats[i]   = VK_FORMAT_UNDEFINED;
                rpk.color_load_ops[i]  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                rpk.color_store_ops[i] = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                rpk.color_layouts[i]   = VK_IMAGE_LAYOUT_UNDEFINED;
            } else {
                auto it = dev->iv_map.find(ca.imageView);
                rpk.color_formats[i]   = (it != dev->iv_map.end())
                                             ? it->second.format
                                             : VK_FORMAT_UNDEFINED;
                rpk.color_load_ops[i]  = ca.loadOp;
                rpk.color_store_ops[i] = ca.storeOp;
                rpk.color_layouts[i]   = ca.imageLayout;
            }
        }
    }

    // Determine depth/stencil attachment.
    // If both depth and stencil point to the same image view (packed format),
    // emit one combined attachment; otherwise use whichever is non-null.
    VkImageView depth_view   = VK_NULL_HANDLE;
    VkImageView stencil_view = VK_NULL_HANDLE;
    if (ri.pDepthAttachment)   depth_view   = ri.pDepthAttachment->imageView;
    if (ri.pStencilAttachment) stencil_view = ri.pStencilAttachment->imageView;

    VkImageView ds_view = (depth_view != VK_NULL_HANDLE) ? depth_view : stencil_view;
    if (ds_view != VK_NULL_HANDLE) {
        {
            std::lock_guard<std::mutex> lk(dev->iv_mtx);
            auto it = dev->iv_map.find(ds_view);
            rpk.depth_format = (it != dev->iv_map.end())
                                   ? it->second.format
                                   : VK_FORMAT_UNDEFINED;
        }
        rpk.depth_layout     = ri.pDepthAttachment   ? ri.pDepthAttachment->imageLayout
                                                      : ri.pStencilAttachment->imageLayout;
        rpk.depth_load_op    = ri.pDepthAttachment   ? ri.pDepthAttachment->loadOp
                                                      : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        rpk.depth_store_op   = ri.pDepthAttachment   ? ri.pDepthAttachment->storeOp
                                                      : VK_ATTACHMENT_STORE_OP_DONT_CARE;
        rpk.stencil_load_op  = ri.pStencilAttachment ? ri.pStencilAttachment->loadOp
                                                      : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        rpk.stencil_store_op = ri.pStencilAttachment ? ri.pStencilAttachment->storeOp
                                                      : VK_ATTACHMENT_STORE_OP_DONT_CARE;
        rpk.stencil_layout   = ri.pStencilAttachment ? ri.pStencilAttachment->imageLayout
                                                      : rpk.depth_layout;
    } else {
        rpk.depth_format     = VK_FORMAT_UNDEFINED;
        rpk.depth_load_op    = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        rpk.depth_store_op   = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        rpk.depth_layout     = VK_IMAGE_LAYOUT_UNDEFINED;
        rpk.stencil_load_op  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        rpk.stencil_store_op = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        rpk.stencil_layout   = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    VkRenderPass rp = get_or_create_rendering_rp(dev, rpk);
    if (rp == VK_NULL_HANDLE) return;

    // Build compact attachment view list and clear values matching the
    // VkAttachmentDescription entries in the render pass (UNDEFINED formats
    // are skipped — they have VK_ATTACHMENT_UNUSED refs in the subpass).
    FramebufferKey fbk = {};
    fbk.render_pass = rp;
    fbk.width       = ri.renderArea.extent.width  + ri.renderArea.offset.x;
    fbk.height      = ri.renderArea.extent.height + ri.renderArea.offset.y;
    fbk.layers      = (ri.viewMask != 0) ? 1 : ri.layerCount;

    VkClearValue clear_values[9];
    uint32_t     att = 0;

    for (uint32_t i = 0; i < rpk.color_count; ++i) {
        if (rpk.color_formats[i] == VK_FORMAT_UNDEFINED) continue;
        fbk.attachments[att] = ri.pColorAttachments[i].imageView;
        clear_values[att]    = ri.pColorAttachments[i].clearValue;
        ++att;
    }
    if (ds_view != VK_NULL_HANDLE) {
        fbk.attachments[att] = ds_view;
        if (ri.pDepthAttachment)
            clear_values[att] = ri.pDepthAttachment->clearValue;
        else
            clear_values[att] = ri.pStencilAttachment->clearValue;
        ++att;
    }
    fbk.attachment_count = att;

    VkFramebuffer fb = get_or_create_fb(dev, fbk);
    if (fb == VK_NULL_HANDLE) return;

    VkRenderPassBeginInfo rpbi = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rpbi.renderPass      = rp;
    rpbi.framebuffer     = fb;
    rpbi.renderArea      = ri.renderArea;
    rpbi.clearValueCount = att;
    rpbi.pClearValues    = clear_values;

    VkSubpassContents contents =
        (ri.flags & VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT)
        ? VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS
        : VK_SUBPASS_CONTENTS_INLINE;

    dev->CmdBeginRenderPass(commandBuffer, &rpbi, contents);
}

// ============================================================================
// CmdEndRenderingKHR
// ============================================================================

static VKAPI_ATTR void VKAPI_CALL layer_CmdEndRenderingKHR(
    VkCommandBuffer commandBuffer)
{
    DeviceData* dev = get_dev(dispatch_key(commandBuffer));
    if (dev) dev->CmdEndRenderPass(commandBuffer);
}

// ============================================================================
// CreateGraphicsPipelines — handle VkPipelineRenderingCreateInfo
// ============================================================================

static VKAPI_ATTR VkResult VKAPI_CALL layer_CreateGraphicsPipelines(
    VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount,
    const VkGraphicsPipelineCreateInfo* pCreateInfos,
    const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines)
{
    DeviceData* dev = get_dev(dispatch_key(device));
    if (!dev) return VK_ERROR_INITIALIZATION_FAILED;

    // Fast path: no pipeline needs modification.
    bool any_dynamic = false;
    for (uint32_t i = 0; i < createInfoCount; ++i) {
        if (pCreateInfos[i].renderPass == VK_NULL_HANDLE &&
            find_pnext(pCreateInfos[i].pNext,
                       VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO) != nullptr) {
            any_dynamic = true;
            break;
        }
    }
    if (!any_dynamic) {
        return dev->CreateGraphicsPipelines(device, pipelineCache, createInfoCount,
                                            pCreateInfos, pAllocator, pPipelines);
    }

    // Process each pipeline individually so we can modify only those that need it.
    VkResult overall = VK_SUCCESS;
    for (uint32_t i = 0; i < createInfoCount; ++i) {
        const VkGraphicsPipelineCreateInfo& orig = pCreateInfos[i];

        if (orig.renderPass != VK_NULL_HANDLE ||
            find_pnext(orig.pNext,
                       VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO) == nullptr) {
            // No dynamic rendering info — pass through unchanged.
            VkResult r = dev->CreateGraphicsPipelines(device, pipelineCache, 1,
                                                       &orig, pAllocator, &pPipelines[i]);
            if (r != VK_SUCCESS && overall == VK_SUCCESS) overall = r;
            continue;
        }

        const auto* prc = static_cast<const VkPipelineRenderingCreateInfo*>(
            find_pnext(orig.pNext, VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO));

        // Build a pipeline-compatible render pass from the format info.
        PipelineRpKey pk = {};
        pk.view_mask   = prc->viewMask;
        pk.color_count = prc->colorAttachmentCount < 8 ? prc->colorAttachmentCount : 8;
        pk.samples     = VK_SAMPLE_COUNT_1_BIT;  // pipelines don't encode sample count here;
                                                  // it comes from VkPipelineMultisampleStateCreateInfo
        for (uint32_t j = 0; j < pk.color_count; ++j)
            pk.color_formats[j] = prc->pColorAttachmentFormats[j];
        pk.depth_format   = prc->depthAttachmentFormat;
        pk.stencil_format = prc->stencilAttachmentFormat;

        // If the pipeline has multisample state, read the actual sample count.
        if (orig.pMultisampleState)
            pk.samples = orig.pMultisampleState->rasterizationSamples;

        VkRenderPass compat_rp = get_or_create_pipeline_rp(dev, pk);
        if (compat_rp == VK_NULL_HANDLE) {
            pPipelines[i] = VK_NULL_HANDLE;
            if (overall == VK_SUCCESS) overall = VK_ERROR_INITIALIZATION_FAILED;
            continue;
        }

        // Build a modified create info: renderPass = our compat RP, subpass = 0,
        // and VkPipelineRenderingCreateInfo removed from the pNext chain.
        // We patch the chain in-place on our local copy; because pNext is a
        // const void* in VkBaseInStructure, we walk using a writable alias.
        VkGraphicsPipelineCreateInfo modci = orig;
        modci.renderPass = compat_rp;
        modci.subpass    = 0;

        // Walk the chain starting at modci.pNext (which is our own field, writable).
        // Find VkPipelineRenderingCreateInfo and unlink it.
        VkBaseInStructure* prev = nullptr;
        VkBaseInStructure* cur  = const_cast<VkBaseInStructure*>(
            static_cast<const VkBaseInStructure*>(modci.pNext));
        while (cur) {
            if (cur->sType == VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO) {
                // Unlink: point predecessor's pNext (or modci.pNext) past cur.
                if (prev == nullptr)
                    modci.pNext = cur->pNext;
                else
                    prev->pNext = cur->pNext;
                // Restore cur->pNext after the call (it's the caller's node).
                const VkBaseInStructure* saved_next = cur->pNext;
                (void)saved_next;  // restored below
                break;
            }
            prev = cur;
            cur  = const_cast<VkBaseInStructure*>(cur->pNext);
        }

        VkResult r = dev->CreateGraphicsPipelines(device, pipelineCache, 1,
                                                   &modci, pAllocator, &pPipelines[i]);

        // Restore the chain linkage in case the caller inspects it afterward.
        if (cur) {
            if (prev == nullptr)
                modci.pNext = cur;
            else
                prev->pNext = cur;
        }

        if (r != VK_SUCCESS && overall == VK_SUCCESS) overall = r;
    }
    return overall;
}

// ============================================================================
// BeginCommandBuffer — handle VkCommandBufferInheritanceRenderingInfo for
// secondary command buffers that will execute inside a dynamic render pass.
// ============================================================================

static VKAPI_ATTR VkResult VKAPI_CALL layer_BeginCommandBuffer(
    VkCommandBuffer commandBuffer, const VkCommandBufferBeginInfo* pBeginInfo)
{
    DeviceData* dev = get_dev(dispatch_key(commandBuffer));
    if (!dev) return VK_ERROR_INITIALIZATION_FAILED;

    if (!(pBeginInfo->flags & VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT) ||
        !pBeginInfo->pInheritanceInfo) {
        return dev->BeginCommandBuffer(commandBuffer, pBeginInfo);
    }

    const VkCommandBufferInheritanceRenderingInfo* iri =
        static_cast<const VkCommandBufferInheritanceRenderingInfo*>(
            find_pnext(pBeginInfo->pInheritanceInfo->pNext,
                       VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO));
    if (!iri) {
        return dev->BeginCommandBuffer(commandBuffer, pBeginInfo);
    }

    // Synthesize a pipeline-compatible render pass matching the inheritance info.
    PipelineRpKey pk = {};
    pk.view_mask   = iri->viewMask;
    pk.color_count = iri->colorAttachmentCount < 8 ? iri->colorAttachmentCount : 8;
    pk.samples     = iri->rasterizationSamples;
    for (uint32_t i = 0; i < pk.color_count; ++i)
        pk.color_formats[i] = iri->pColorAttachmentFormats[i];
    pk.depth_format   = iri->depthAttachmentFormat;
    pk.stencil_format = iri->stencilAttachmentFormat;

    VkRenderPass rp = get_or_create_pipeline_rp(dev, pk);
    if (rp == VK_NULL_HANDLE) {
        return dev->BeginCommandBuffer(commandBuffer, pBeginInfo);
    }

    // Build a modified inheritance info with our render pass, subpass = 0.
    VkCommandBufferInheritanceInfo inh = *pBeginInfo->pInheritanceInfo;
    inh.renderPass = rp;
    inh.subpass    = 0;

    // Remove VkCommandBufferInheritanceRenderingInfo from inh.pNext.
    // inh is our local copy so inh.pNext is writable directly.
    VkBaseInStructure* iprev = nullptr;
    VkBaseInStructure* icur  = const_cast<VkBaseInStructure*>(
        static_cast<const VkBaseInStructure*>(inh.pNext));
    while (icur) {
        if (icur->sType == VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO) {
            if (iprev == nullptr)
                inh.pNext = icur->pNext;
            else
                iprev->pNext = icur->pNext;
            break;
        }
        iprev = icur;
        icur  = const_cast<VkBaseInStructure*>(icur->pNext);
    }

    VkCommandBufferBeginInfo modbi = *pBeginInfo;
    modbi.pInheritanceInfo = &inh;

    VkResult r = dev->BeginCommandBuffer(commandBuffer, &modbi);

    // Restore chain (inh is local but icur points into the caller's nodes).
    if (icur) {
        if (iprev == nullptr)
            inh.pNext = icur;
        else
            iprev->pNext = icur;
    }

    return r;
}

// ============================================================================
// GetPhysicalDeviceFeatures2 — advertise dynamicRendering = TRUE
// ============================================================================

static VKAPI_ATTR void VKAPI_CALL layer_GetPhysicalDeviceFeatures2(
    VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures2* pFeatures)
{
    InstanceData* inst = get_inst(dispatch_key(physicalDevice));
    if (!inst || !inst->GetPhysicalDeviceFeatures2) return;
    inst->GetPhysicalDeviceFeatures2(physicalDevice, pFeatures);

    // Walk pNext and set dynamicRendering = VK_TRUE wherever the app asked.
    VkBaseOutStructure* p = reinterpret_cast<VkBaseOutStructure*>(pFeatures->pNext);
    while (p) {
        if (p->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES) {
            reinterpret_cast<VkPhysicalDeviceDynamicRenderingFeatures*>(p)->dynamicRendering = VK_TRUE;
        }
        p = p->pNext;
    }
}

// ============================================================================
// Instance lifecycle
// ============================================================================

static VKAPI_ATTR VkResult VKAPI_CALL layer_CreateInstance(
    const VkInstanceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkInstance* pInstance)
{
    VkLayerInstanceCreateInfo* chain = get_instance_chain_info(pCreateInfo, VK_LAYER_LINK_INFO);
    if (!chain) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr next_gipa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;

    auto* next_create = reinterpret_cast<PFN_vkCreateInstance>(
        next_gipa(nullptr, "vkCreateInstance"));
    VkResult r = next_create(pCreateInfo, pAllocator, pInstance);
    if (r != VK_SUCCESS) return r;

    InstanceData* data = new InstanceData();
    data->instance    = *pInstance;
    data->next_gipa   = next_gipa;

#define LOAD_INST(name) \
    data->name = reinterpret_cast<PFN_vk##name>(next_gipa(*pInstance, "vk" #name))

    LOAD_INST(GetPhysicalDeviceFeatures2);
    if (!data->GetPhysicalDeviceFeatures2)
        data->GetPhysicalDeviceFeatures2 =
            reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
                next_gipa(*pInstance, "vkGetPhysicalDeviceFeatures2KHR"));
    LOAD_INST(EnumerateDeviceExtensionProperties);

#undef LOAD_INST

    std::lock_guard<std::mutex> lk(g_inst_map_mtx);
    g_inst_map[dispatch_key(*pInstance)] = data;
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL layer_DestroyInstance(
    VkInstance instance, const VkAllocationCallbacks* pAllocator)
{
    void* key = dispatch_key(instance);
    InstanceData* data;
    {
        std::lock_guard<std::mutex> lk(g_inst_map_mtx);
        auto it = g_inst_map.find(key);
        if (it == g_inst_map.end()) return;
        data = it->second;
        g_inst_map.erase(it);
    }
    auto* next_destroy = reinterpret_cast<PFN_vkDestroyInstance>(
        data->next_gipa(instance, "vkDestroyInstance"));
    if (next_destroy) next_destroy(instance, pAllocator);
    delete data;
}

// ============================================================================
// Device lifecycle
// ============================================================================

static VKAPI_ATTR VkResult VKAPI_CALL layer_CreateDevice(
    VkPhysicalDevice physicalDevice,
    const VkDeviceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkDevice* pDevice)
{
    VkLayerDeviceCreateInfo* chain = get_device_chain_info(pCreateInfo, VK_LAYER_LINK_INFO);
    if (!chain) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr next_gipa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr   next_gdpa = chain->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;

    // Strip VK_KHR_dynamic_rendering from the extension list so the ICD does
    // not see an extension it does not support.
    const char* strip = "VK_KHR_dynamic_rendering";
    uint32_t orig_n   = pCreateInfo->enabledExtensionCount;
    const char** new_exts = new const char*[orig_n ? orig_n : 1];
    uint32_t new_n = 0;
    for (uint32_t i = 0; i < orig_n; ++i) {
        if (strcmp(pCreateInfo->ppEnabledExtensionNames[i], strip) != 0)
            new_exts[new_n++] = pCreateInfo->ppEnabledExtensionNames[i];
    }

    VkDeviceCreateInfo modci = *pCreateInfo;
    modci.enabledExtensionCount   = new_n;
    modci.ppEnabledExtensionNames = new_exts;

    // Strip VkPhysicalDeviceDynamicRenderingFeatures from the pNext chain.
    // We find the node, unlink it, call CreateDevice, then restore the chain.
    VkBaseInStructure* dr_prev = nullptr;
    VkBaseInStructure* dr_node = nullptr;
    {
        VkBaseInStructure* prev = nullptr;
        VkBaseInStructure* cur  = const_cast<VkBaseInStructure*>(
            static_cast<const VkBaseInStructure*>(modci.pNext));
        while (cur) {
            if (cur->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES) {
                dr_node = cur;
                dr_prev = prev;
                if (prev == nullptr)
                    modci.pNext = cur->pNext;
                else
                    prev->pNext = cur->pNext;
                break;
            }
            prev = cur;
            cur  = const_cast<VkBaseInStructure*>(cur->pNext);
        }
    }

    auto* next_create = reinterpret_cast<PFN_vkCreateDevice>(
        next_gipa(VK_NULL_HANDLE, "vkCreateDevice"));
    VkResult r = next_create(physicalDevice, &modci, pAllocator, pDevice);

    // Restore the pNext chain (the structs belong to the caller).
    if (dr_node) {
        if (dr_prev == nullptr)
            modci.pNext = dr_node;
        else
            dr_prev->pNext = dr_node;
    }

    delete[] new_exts;

    if (r != VK_SUCCESS) return r;

    DeviceData* data = new DeviceData();
    data->device   = *pDevice;
    data->next_gdpa = next_gdpa;

#define LOAD_DEV(name) \
    data->name = reinterpret_cast<PFN_vk##name>(next_gdpa(*pDevice, "vk" #name))

    LOAD_DEV(CreateRenderPass);
    LOAD_DEV(DestroyRenderPass);
    LOAD_DEV(CreateFramebuffer);
    LOAD_DEV(DestroyFramebuffer);
    LOAD_DEV(CmdBeginRenderPass);
    LOAD_DEV(CmdEndRenderPass);
    LOAD_DEV(CreateGraphicsPipelines);
    LOAD_DEV(CreateImage);
    LOAD_DEV(DestroyImage);
    LOAD_DEV(CreateImageView);
    LOAD_DEV(DestroyImageView);
    LOAD_DEV(BeginCommandBuffer);

#undef LOAD_DEV

    std::lock_guard<std::mutex> lk(g_dev_map_mtx);
    g_dev_map[dispatch_key(*pDevice)] = data;
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL layer_DestroyDevice(
    VkDevice device, const VkAllocationCallbacks* pAllocator)
{
    void* key = dispatch_key(device);
    DeviceData* data;
    {
        std::lock_guard<std::mutex> lk(g_dev_map_mtx);
        auto it = g_dev_map.find(key);
        if (it == g_dev_map.end()) return;
        data = it->second;
        g_dev_map.erase(it);
    }

    // Destroy all cached render passes and framebuffers.
    {
        std::lock_guard<std::mutex> lk(data->cache_mtx);
        for (auto& kv : data->rendering_rp_cache)
            data->DestroyRenderPass(device, kv.second, nullptr);
        for (auto& kv : data->pipeline_rp_cache)
            data->DestroyRenderPass(device, kv.second, nullptr);
        for (auto& kv : data->fb_cache)
            data->DestroyFramebuffer(device, kv.second, nullptr);
    }

    auto* next_destroy = reinterpret_cast<PFN_vkDestroyDevice>(
        data->next_gdpa(device, "vkDestroyDevice"));
    if (next_destroy) next_destroy(device, pAllocator);
    delete data;
}

// ============================================================================
// Entrypoint dispatch tables
// ============================================================================

static PFN_vkVoidFunction intercept_instance(const char* name) {
#define M(fn) if (!strcmp(name, "vk" #fn)) return reinterpret_cast<PFN_vkVoidFunction>(layer_##fn)
    M(CreateInstance);
    M(DestroyInstance);
    M(CreateDevice);
    M(GetPhysicalDeviceFeatures2);
#undef M
    // Core 1.3 alias for GetPhysicalDeviceFeatures2
    if (!strcmp(name, "vkGetPhysicalDeviceFeatures2KHR"))
        return reinterpret_cast<PFN_vkVoidFunction>(layer_GetPhysicalDeviceFeatures2);
    return nullptr;
}

static PFN_vkVoidFunction intercept_device(const char* name) {
#define M(fn) if (!strcmp(name, "vk" #fn)) return reinterpret_cast<PFN_vkVoidFunction>(layer_##fn)
    M(DestroyDevice);
    M(CreateImage);
    M(DestroyImage);
    M(CreateImageView);
    M(DestroyImageView);
    M(CreateGraphicsPipelines);
    M(BeginCommandBuffer);
#undef M
    // Both core 1.3 and KHR aliases for dynamic rendering commands.
    if (!strcmp(name, "vkCmdBeginRenderingKHR") || !strcmp(name, "vkCmdBeginRendering"))
        return reinterpret_cast<PFN_vkVoidFunction>(layer_CmdBeginRenderingKHR);
    if (!strcmp(name, "vkCmdEndRenderingKHR") || !strcmp(name, "vkCmdEndRendering"))
        return reinterpret_cast<PFN_vkVoidFunction>(layer_CmdEndRenderingKHR);
    return nullptr;
}

// ============================================================================
// Exported entry points (visible symbols in the .so / .dll)
// ============================================================================

extern "C" {

VEL_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetInstanceProcAddr(VkInstance instance, const char* pName) {
    PFN_vkVoidFunction fn = intercept_instance(pName);
    if (fn) return fn;
    fn = intercept_device(pName);
    if (fn) return fn;
    if (!instance) return nullptr;
    InstanceData* data = get_inst(dispatch_key(instance));
    if (!data) return nullptr;
    return data->next_gipa(instance, pName);
}

VEL_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetDeviceProcAddr(VkDevice device, const char* pName) {
    PFN_vkVoidFunction fn = intercept_device(pName);
    if (fn) return fn;
    if (!device) return nullptr;
    DeviceData* data = get_dev(dispatch_key(device));
    if (!data) return nullptr;
    return data->next_gdpa(device, pName);
}

VEL_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
vkEnumerateInstanceExtensionProperties(
    const char* pLayerName, uint32_t* pPropertyCount,
    VkExtensionProperties* pProperties)
{
    // Only expose when queried for our specific layer name.
    if (pLayerName && !strcmp(pLayerName, "VK_LAYER_KHRONOS_dynamic_rendering")) {
        static const VkExtensionProperties exts[] = {};  // no instance extensions
        uint32_t n = sizeof(exts) / sizeof(exts[0]);
        if (!pProperties) { *pPropertyCount = n; return VK_SUCCESS; }
        uint32_t out = (*pPropertyCount < n) ? *pPropertyCount : n;
        memcpy(pProperties, exts, out * sizeof(VkExtensionProperties));
        *pPropertyCount = out;
        return out < n ? VK_INCOMPLETE : VK_SUCCESS;
    }
    return VK_ERROR_LAYER_NOT_PRESENT;
}

VEL_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
vkEnumerateInstanceLayerProperties(
    uint32_t* pPropertyCount, VkLayerProperties* pProperties)
{
    static const VkLayerProperties layer = {
        "VK_LAYER_KHRONOS_dynamic_rendering", VK_API_VERSION_1_1, 1,
        "Khronos dynamic rendering emulation layer"
    };
    if (!pProperties) { *pPropertyCount = 1; return VK_SUCCESS; }
    if (*pPropertyCount < 1) { *pPropertyCount = 0; return VK_INCOMPLETE; }
    *pProperties    = layer;
    *pPropertyCount = 1;
    return VK_SUCCESS;
}

VEL_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
vkEnumerateDeviceExtensionProperties(
    VkPhysicalDevice physicalDevice, const char* pLayerName,
    uint32_t* pPropertyCount, VkExtensionProperties* pProperties)
{
    static const VkExtensionProperties provided = {
        VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME, VK_KHR_DYNAMIC_RENDERING_SPEC_VERSION
    };

    if (pLayerName && !strcmp(pLayerName, "VK_LAYER_KHRONOS_dynamic_rendering")) {
        // Direct layer query — just return our one extension.
        if (!pProperties) { *pPropertyCount = 1; return VK_SUCCESS; }
        if (*pPropertyCount < 1) { *pPropertyCount = 0; return VK_INCOMPLETE; }
        *pProperties    = provided;
        *pPropertyCount = 1;
        return VK_SUCCESS;
    }

    if (!physicalDevice) return VK_ERROR_LAYER_NOT_PRESENT;

    InstanceData* inst = get_inst(dispatch_key(physicalDevice));
    if (!inst || !inst->EnumerateDeviceExtensionProperties)
        return VK_ERROR_INITIALIZATION_FAILED;

    // Query the driver's extensions first.
    uint32_t driver_count = 0;
    VkResult r = inst->EnumerateDeviceExtensionProperties(
        physicalDevice, nullptr, &driver_count, nullptr);
    if (r != VK_SUCCESS) return r;

    VkExtensionProperties* driver_exts = new VkExtensionProperties[driver_count ? driver_count : 1];
    r = inst->EnumerateDeviceExtensionProperties(
        physicalDevice, nullptr, &driver_count, driver_exts);
    if (r != VK_SUCCESS) { delete[] driver_exts; return r; }

    // Check whether the driver already exposes dynamic rendering natively.
    bool already_present = false;
    for (uint32_t i = 0; i < driver_count; ++i) {
        if (!strcmp(driver_exts[i].extensionName, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME)) {
            already_present = true;
            break;
        }
    }

    uint32_t total = driver_count + (already_present ? 0 : 1);

    if (!pProperties) {
        *pPropertyCount = total;
        delete[] driver_exts;
        return VK_SUCCESS;
    }

    uint32_t out = (*pPropertyCount < total) ? *pPropertyCount : total;
    uint32_t copy = (out < driver_count) ? out : driver_count;
    memcpy(pProperties, driver_exts, copy * sizeof(VkExtensionProperties));
    if (!already_present && copy < out) {
        pProperties[copy] = provided;
        copy++;
    }
    *pPropertyCount = copy;
    delete[] driver_exts;
    return copy < total ? VK_INCOMPLETE : VK_SUCCESS;
}

VEL_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
vkEnumerateDeviceLayerProperties(
    VkPhysicalDevice, uint32_t* pPropertyCount, VkLayerProperties* pProperties)
{
    static const VkLayerProperties layer = {
        "VK_LAYER_KHRONOS_dynamic_rendering", VK_API_VERSION_1_1, 1,
        "Khronos dynamic rendering emulation layer"
    };
    if (!pProperties) { *pPropertyCount = 1; return VK_SUCCESS; }
    if (*pPropertyCount < 1) { *pPropertyCount = 0; return VK_INCOMPLETE; }
    *pProperties    = layer;
    *pPropertyCount = 1;
    return VK_SUCCESS;
}

VEL_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* pVersionStruct) {
    if (pVersionStruct->loaderLayerInterfaceVersion < 2)
        return VK_ERROR_INITIALIZATION_FAILED;
    pVersionStruct->loaderLayerInterfaceVersion = 2;
    pVersionStruct->pfnGetInstanceProcAddr      = vkGetInstanceProcAddr;
    pVersionStruct->pfnGetDeviceProcAddr        = vkGetDeviceProcAddr;
    pVersionStruct->pfnGetPhysicalDeviceProcAddr = nullptr;
    return VK_SUCCESS;
}

}  // extern "C"
