////////////////////////////////////////////////////////////////////////////////
//
// Demo framework for the Vookoo for the Vookoo high level C++ Vulkan interface.
//
// (C) Andy Thomason 2017 MIT License
//
// This is an optional demo framework for the Vookoo high level C++ Vulkan interface.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef VKU_FRAMEWORK_HPP
#define VKU_FRAMEWORK_HPP

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef VC_EXTRALEAN
#define VC_EXTRALEAN
#endif

#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#define GLFW_EXPOSE_NATIVE_WIN32
#define VKU_SURFACE "VK_KHR_win32_surface"
#pragma warning(disable : 4005)
#else
#define VK_USE_PLATFORM_XLIB_KHR
#define GLFW_EXPOSE_NATIVE_X11
#define VKU_SURFACE "VK_KHR_xlib_surface"
#endif
#define VK_MAJOR_VERSION 1
#define VK_MINOR_VERSION 1

#define VK_NO_PROTOTYPES
#include "volk/volk.h"

#ifndef VKU_NO_GLFW
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#endif

// Undo damage done by windows.h
#undef APIENTRY
#undef None
#undef max
#undef min

#include <array>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <thread>
#include <chrono>
#include <functional>
#include <cstddef>
#include <set>

#include <vulkan/vulkan.hpp>
#include <vku/vku.hpp>
#include <vku/vku_addon.hpp>
#include <tbb/parallel_invoke.h>

namespace vku {

	static constexpr uint32_t const
		MAX_NUM_DESCRIPTOR_SETS = 32,
		MAX_NUM_UNIFORM_BUFFERS = 8,
		MAX_NUM_IMAGES = 32,
		MAX_NUM_STORAGE_BUFFERS = 8;

	// for avoiding lamda heap
	typedef struct {
		vk::CommandBuffer& __restrict cb;
	} const scattered_offscreen_renderpass;
	typedef void(*const scattered_offscreen_renderpass_function)(scattered_offscreen_renderpass const& __restrict);
	typedef void(*scattered_offscreen_renderpass_function_unconst)(scattered_offscreen_renderpass const& __restrict);

	typedef struct {
		vk::CommandBuffer& __restrict cb_transfer;
		vk::CommandBuffer& __restrict cb_render;
		bool const IsDirty;
	} const compute_gpu_function;
	typedef bool const(*const compute_function)(compute_gpu_function const& __restrict);

	typedef struct {
		vk::CommandBuffer& __restrict cb;
		vk::UniqueEvent& __restrict event;
		vk::RenderPassBeginInfo const& __restrict rpbi;
	} const static_renderpass;
	typedef void(*const static_renderpass_function)(static_renderpass const& __restrict);
	typedef void(*static_renderpass_function_unconst)(static_renderpass const& __restrict);

	typedef struct {
		vk::CommandBuffer& __restrict cb;
		bool const dma_transfer_enabled;	// when true opacity volume is malplped8 ulpd8ated8 and8 ulplpoad8ed8
		
	} const dynamic_renderpass;
	typedef void(*const dynamic_renderpass_function)(dynamic_renderpass const& __restrict);

	typedef struct {
		vk::CommandBuffer& __restrict cb_transfer;
		vk::CommandBuffer& __restrict cb_render;
		vk::UniqueEvent* const& __restrict events;
		vk::RenderPassBeginInfo const rpbi;
		
	} const overlay_renderpass;
	typedef void(*const overlay_renderpass_function)(overlay_renderpass const& __restrict);

/// This class provides an optional interface to the vulkan instance, devices and queues.
/// It is not used by any of the other classes directly and so can be safely ignored if Vookoo
/// is embedded in an engine.
/// See https://vulkan-tutorial.com for details of many operations here.
class Framework {
public:
  Framework() {
  }

  // Construct a framework containing the instance, a device and one or more queues.
  void FrameworkCreate(const std::string &name) {
	  uint32_t  const apiVersion(VK_MAKE_VERSION(VK_MAJOR_VERSION, VK_MINOR_VERSION, 0));
    vku::InstanceMaker im{};
    im.defaultLayers();
	im.applicationName(name.c_str());
	im.engineName("supersinfulsilicon");
	im.applicationVersion(1);
	im.engineVersion(1);
	im.apiVersion(apiVersion);

    instance_ = im.createUnique();

#ifndef NDEBUG
    callback_ = DebugCallback(*instance_);
#endif

    auto const pds = instance_->enumeratePhysicalDevices();
	for (auto const& i : pds)
	{
		uint32_t const physicalDeviceApiVersion = i.getProperties().apiVersion;
		if (physicalDeviceApiVersion >= apiVersion) {
			physical_device_ = i;
			fmt::print(fg(fmt::color::magenta),  "[ Vulkan {:d}.{:d} ]" "\n", VK_VERSION_MAJOR(physicalDeviceApiVersion), 
																		      VK_VERSION_MINOR(physicalDeviceApiVersion));
			fmt::print(fg(fmt::color::white),   "[ {:s} ]" "\n", i.getProperties().deviceName);
			break;
		}
	}
	if (nullptr == physical_device_) {
		fmt::print(fg(fmt::color::red),   "[ ! Vulkan 1.1 - Not supported by any gpu device ! ]" "\n");
	}
    auto qprops = physical_device_.getQueueFamilyProperties();
    
    graphicsQueueFamilyIndex_ = 0;
    computeQueueFamilyIndex_ = 0;
	transferQueueFamilyIndex_ = 0;
	vk::QueueFlags  
		searchGraphics = vk::QueueFlagBits::eGraphics,
		searchCompute = vk::QueueFlagBits::eGraphics|vk::QueueFlagBits::eCompute,  // **************** any ShaderReadOnlyOptimal (not general)sampler access in compute shader requires graphics queue aswell
		searchTransfer = vk::QueueFlagBits::eTransfer; // speedy 8x8 granularity (multiple of 8) transfer queue
																		// ** resolution must be divisible by 8 (all normally are)
    // Look for an omnipurpose queue family first
    // It is better if we can schedule operations without barriers and semaphores.
    // The Spec says: "If an implementation exposes any queue family that supports graphics operations,
    // at least one queue family of at least one physical device exposed by the implementation
    // must support both graphics and compute operations."
    // Also: All commands that are allowed on a queue that supports transfer operations are
    // also allowed on a queue that supports either graphics or compute operations...
    // As a result we can expect a queue family with at least all three and maybe all four modes.

	std::set<uint32_t> enabledQueueIndices;
			
    for (int32_t qi = (int32_t)qprops.size() - 1; qi >= 0; --qi) {	// start from back to capture unique queues first
      auto &qprop = qprops[qi];

      if (searchGraphics && (qprop.queueFlags & searchGraphics) == searchGraphics) {
			graphicsQueueFamilyIndex_ = qi;
			enabledQueueIndices.emplace(qi);
			if (0 == qi) {
				searchGraphics = (vk::QueueFlagBits)0; // prevent further search only if equal to zero for graphics queue (default index)
			}
			FMT_LOG_OK(GPU_LOG, "Graphics Queue Selected < {:s} >", vk::to_string(qprop.queueFlags));
      }
	  if (searchCompute && (qprop.queueFlags & searchCompute) == searchCompute) {
		  computeQueueFamilyIndex_ = qi;
		  enabledQueueIndices.emplace(qi);
		  searchCompute = (vk::QueueFlagBits)0; // prevent further search
		  FMT_LOG_OK(GPU_LOG, "Compute Queue Selected < {:s} >", vk::to_string(qprop.queueFlags)); 
	  }
	  if (searchTransfer && (qprop.queueFlags & searchTransfer) == searchTransfer) {
		  enabledQueueIndices.emplace(qi);
		  transferQueueFamilyIndex_ = qi;
		  searchTransfer = (vk::QueueFlagBits)0;
		  FMT_LOG(GPU_LOG, "Present Queue Pending < {:s} >\n", vk::to_string(qprop.queueFlags));
	  }
    }

    memprops_ = physical_device_.getMemoryProperties();

	vk::PhysicalDeviceFeatures const supportedFeatures = physical_device_.getFeatures();
	vk::PhysicalDeviceFeatures enabledFeatures;
	enabledFeatures.geometryShader = supportedFeatures.geometryShader;
	enabledFeatures.samplerAnisotropy = supportedFeatures.samplerAnisotropy;
	enabledFeatures.independentBlend = supportedFeatures.independentBlend;
	//enabledFeatures.robustBufferAccess = supportedFeatures.robustBufferAccess; // safer but a lot slower good for debugging out of bounds access
	enabledFeatures.textureCompressionBC = supportedFeatures.textureCompressionBC;
	enabledFeatures.shaderStorageImageExtendedFormats = supportedFeatures.shaderStorageImageExtendedFormats;

	PRINT_FEATURE(enabledFeatures.geometryShader, "geometry shader");
	PRINT_FEATURE(enabledFeatures.samplerAnisotropy, "anisotropic filtering");
	PRINT_FEATURE(enabledFeatures.independentBlend, "independent blending"); // this feautre enables distinct blend pieline attachments to be used (important!)
	//PRINT_FEATURE(enabledFeatures.robustBufferAccess, "robust buffer access");  // this feature enabled driver related bounds checking on buffers
	PRINT_FEATURE(enabledFeatures.textureCompressionBC, "texture compression");	  // which prevents corruption/termination of application if out of bounds access' happen
	PRINT_FEATURE(enabledFeatures.shaderStorageImageExtendedFormats, "extended compute image formats");	  // which prevents corruption/termination of application if out of bounds access' happen

    // todo: find optimal texture format
    // auto rgbaprops = physical_device_.getFormatProperties(vk::Format::eR8G8B8A8Unorm);

    vku::DeviceMaker dm{};
    dm.defaultLayers();

	// add extensions
	auto const extensions = physical_device_.enumerateDeviceExtensionProperties();

	ADD_EXTENSION(extensions, dm, VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
	ADD_EXTENSION(extensions, dm, VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);

    dm.queue(graphicsQueueFamilyIndex_); // Add graphics queue as first queue for device
	enabledQueueIndices.erase(graphicsQueueFamilyIndex_); // remove from rest of enabled queue indices
	// pre-liminary enable extra queues (compute and transfer/present)
	for (auto const queueIndexEnabled : enabledQueueIndices) {
		dm.queue(queueIndexEnabled);
	}	

    device_ = dm.createUnique(physical_device_, enabledFeatures);
    
    vk::PipelineCacheCreateInfo pipelineCacheInfo{};
    pipelineCache_ = device_->createPipelineCacheUnique(pipelineCacheInfo);

    std::vector<vk::DescriptorPoolSize> poolSizes;
    poolSizes.emplace_back(vk::DescriptorType::eUniformBuffer, MAX_NUM_UNIFORM_BUFFERS);
    poolSizes.emplace_back(vk::DescriptorType::eCombinedImageSampler, MAX_NUM_IMAGES);
    poolSizes.emplace_back(vk::DescriptorType::eStorageBuffer, MAX_NUM_STORAGE_BUFFERS);

    // Create an arbitrary number of descriptors in a pool.
    // Allow the descriptors to be freed, possibly not optimal behaviour.
    vk::DescriptorPoolCreateInfo descriptorPoolInfo{};
	//descriptorPoolInfo.flags = // vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    descriptorPoolInfo.maxSets = MAX_NUM_DESCRIPTOR_SETS;
    descriptorPoolInfo.poolSizeCount = (uint32_t)poolSizes.size();
    descriptorPoolInfo.pPoolSizes = poolSizes.data();
    descriptorPool_ = device_->createDescriptorPoolUnique(descriptorPoolInfo);

    ok_ = true;
  }

  void dumpCaps(std::ostream &os) const {
    os << "Memory Types\n";
    for (uint32_t i = 0; i != memprops_.memoryTypeCount; ++i) {
      os << "  type" << i << " heap" << memprops_.memoryTypes[i].heapIndex << " " << vk::to_string(memprops_.memoryTypes[i].propertyFlags) << "\n";
    }
    os << "Heaps\n";
    for (uint32_t i = 0; i != memprops_.memoryHeapCount; ++i) {
      os << "  heap" << vk::to_string(memprops_.memoryHeaps[i].flags) << " " << memprops_.memoryHeaps[i].size << "\n";
    }
  }

  /// Get the Vulkan instance.
  const vk::Instance instance() const { return *instance_; }

  /// Get the Vulkan device.
  const vk::Device device() const { return *device_; }

  /// Get the queue used to submit graphics jobs
  const vk::Queue graphicsQueue() const { return device_->getQueue(graphicsQueueFamilyIndex_, 0); }

  /// Get the queue used to submit compute jobs
  const vk::Queue computeQueue() const { return device_->getQueue(computeQueueFamilyIndex_, 0); }

  // Get the queue used to transfer data
  const vk::Queue transferQueue() const { return device_->getQueue(transferQueueFamilyIndex_, 0); }

  /// Get the physical device.
  const vk::PhysicalDevice &physicalDevice() const { return physical_device_; }

  /// Get the default pipeline cache (you can use your own if you like).
  const vk::PipelineCache pipelineCache() const { return *pipelineCache_; }

  /// Get the default descriptor pool (you can use your own if you like).
  const vk::DescriptorPool descriptorPool() const { return *descriptorPool_; }

  /// Get the family index for the graphics queues.
  uint32_t graphicsQueueFamilyIndex() const { return graphicsQueueFamilyIndex_; }

  /// Get the family index for the compute queues.
  uint32_t computeQueueFamilyIndex() const { return computeQueueFamilyIndex_; }

  /// Get the family index for the compute queues.
  uint32_t transferQueueFamilyIndex() const { return transferQueueFamilyIndex_; }

  const vk::PhysicalDeviceMemoryProperties &memprops() const { return memprops_; }

  /// Clean up the framework satisfying the Vulkan verification layers.
  ~Framework() {
    if (device_) {
      device_->waitIdle();
      if (pipelineCache_) {
        pipelineCache_.reset();
      }
      if (descriptorPool_) {
        descriptorPool_.reset();
      }
      device_.reset();
    }

    if (instance_) {
#ifndef NDEBUG
      callback_.reset();
#endif
      instance_.reset();
    }
  }

  Framework &operator=(Framework &&rhs) = default;

  /// Returns true if the Framework has been built correctly.
  bool ok() const { return ok_; }

private:
  vk::UniqueInstance instance_;
#ifndef NDEBUG
  vku::DebugCallback callback_;
#endif
  vk::UniqueDevice device_;
  //vk::DebugReportCallbackEXT callback_;
  vk::PhysicalDevice physical_device_;
  vk::UniquePipelineCache pipelineCache_;
  vk::UniqueDescriptorPool descriptorPool_;
  uint32_t graphicsQueueFamilyIndex_;
  uint32_t computeQueueFamilyIndex_;
  uint32_t transferQueueFamilyIndex_;
  vk::PhysicalDeviceMemoryProperties memprops_;

  bool ok_ = false;
};

/// This class wraps a window, a surface and a swap chain for that surface.
BETTER_ENUM(eCommandPools, uint32_t const, DEFAULT_POOL = 0, OVERLAY_POOL, SCATTERED_POOL, TRANSIENT_POOL, DMA_TRANSFER_POOL_THREAD_PRIMARY, DMA_TRANSFER_POOL_THREAD_SECONDARY, COMPUTE_POOL);
BETTER_ENUM(eGraphicsEvent, uint32_t const, RESERVED_PLACEHOLDER = 0, RENDERED_STATIC);
BETTER_ENUM(eFrameBuffers, uint32_t const, COLOR_DEPTH, COLOR_ONLY);
BETTER_ENUM(eOverlayBuffers, uint32_t const, TRANSFER, RENDER);
BETTER_ENUM(eComputeBuffers, uint32_t const, TRANSFER, COMPUTE);
BETTER_ENUM(eComputeState, int32_t const, STALE = -1, TRANSFERED = 0, COMPUTED = 1);

class Window {

public:
  Window() {
  }

#ifndef VKU_NO_GLFW
  /// Construct a window, surface and swapchain using a GLFW window.
  Window(const vk::Instance &instance, const vk::Device &device, const vk::PhysicalDevice &physicalDevice, uint32_t const graphicsQueueFamilyIndex, uint32_t const computeQueueFamilyIndex, GLFWwindow *window) {
#ifdef VK_USE_PLATFORM_WIN32_KHR
    auto module = GetModuleHandle(nullptr);
    auto handle = glfwGetWin32Window(window);
	glfwSetWindowUserPointer(window, this);
    auto ci = vk::Win32SurfaceCreateInfoKHR{{}, module, handle};
    auto surface = instance.createWin32SurfaceKHR(ci);
#endif
#ifdef VK_USE_PLATFORM_XLIB_KHR
    auto display = glfwGetX11Display();
    auto x11window = glfwGetX11Window(window);
    auto ci = vk::XlibSurfaceCreateInfoKHR{{}, display, x11window};
    auto surface = instance.createXlibSurfaceKHR(ci);
#endif
    init(instance, device, physicalDevice, graphicsQueueFamilyIndex, computeQueueFamilyIndex, surface);
  }
#endif

  Window(const vk::Instance &instance, const vk::Device &device, const vk::PhysicalDevice &physicalDevice, uint32_t const graphicsQueueFamilyIndex, uint32_t const computeQueueFamilyIndex, vk::SurfaceKHR surface) {
    init(instance, device, physicalDevice, graphicsQueueFamilyIndex, computeQueueFamilyIndex, surface);
  }

  void init(const vk::Instance &instance, const vk::Device &device, const vk::PhysicalDevice &physicalDevice, uint32_t const graphicsQueueFamilyIndex, uint32_t const computeQueueFamilyIndex, vk::SurfaceKHR surface) {
	  //surface_ = vk::UniqueSurfaceKHR(surface);
	  //surface_ = vk::UniqueSurfaceKHR(surface, vk::SurfaceKHRDeleter{ instance });
	  surface_ = surface;
	  instance_ = instance;
	  device_ = device;
	  presentQueueFamily_ = 0;
	  auto &pd = physicalDevice;
	  auto qprops = pd.getQueueFamilyProperties();

	  // start from back to capture unique present transfer optimized queue (DMA)
	  // otherwise default to queue 0 (default) if surface is unsupported
	  for (int32_t qi = (int32_t)qprops.size() - 1; qi >= 0 ; --qi) {
		  auto &qprop = qprops[qi];
		  if (pd.getSurfaceSupportKHR(qi, surface_) && (qprop.queueFlags & vk::QueueFlagBits::eTransfer) == vk::QueueFlagBits::eTransfer) {
			  
			  presentQueueFamily_ = qi;
			  FMT_LOG_OK(GPU_LOG, "Present Queue Selected < {:s} >", vk::to_string(qprop.queueFlags));
			  break;
		  }
	  }
	  // will default to queue 0 if unsupported
	  uint32_t const transferQueueFamilyIndex = presentQueueFamily_;

	  auto fmts = pd.getSurfaceFormatsKHR(surface_);
	  swapchainImageFormat_ = fmts[0].format;
	  swapchainColorSpace_ = fmts[0].colorSpace;
	  if (fmts.size() == 1 && swapchainImageFormat_ == vk::Format::eUndefined) {
		  swapchainImageFormat_ = vk::Format::eB8G8R8A8Unorm;
		  swapchainColorSpace_ = vk::ColorSpaceKHR::eSrgbNonlinear;
	  }
	  else {
		  for (auto &fmt : fmts) {
			  if (fmt.format == vk::Format::eB8G8R8A8Unorm && fmt.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
				  swapchainImageFormat_ = fmt.format;
				  swapchainColorSpace_ = fmt.colorSpace;
				  break;
			  }
		  }
	  }
	  if (swapchainImageFormat_ == vk::Format::eB8G8R8A8Unorm && swapchainColorSpace_ == vk::ColorSpaceKHR::eSrgbNonlinear) {
		  fmt::print(fg(fmt::color::lime_green), "32bit SRGB Backbuffer" "\n");
	  }
	  else {
		  fmt::print(fg(fmt::color::red), "[FAIL] 32bit NON-SRGB Backbuffer" "\n");
	  }

	  auto surfaceCaps = pd.getSurfaceCapabilitiesKHR(surface_);
	  width_ = surfaceCaps.currentExtent.width;
	  height_ = surfaceCaps.currentExtent.height;

	  auto pms = pd.getSurfacePresentModesKHR(surface_);
	  vk::PresentModeKHR presentMode = pms[0];
	  if (std::find(pms.begin(), pms.end(), vk::PresentModeKHR::eFifo) != pms.end()) {
		  presentMode = vk::PresentModeKHR::eFifo;
	  }
	  else {
		  std::cout << "No fifo mode available\n";
		  return;
	  }

	  //std::cout << "using " << vk::to_string(presentMode) << "\n";

	  vk::SwapchainCreateInfoKHR swapinfo{};
	  std::array<uint32_t, 2> queueFamilyIndices = { graphicsQueueFamilyIndex, presentQueueFamily_ };
	  bool sameQueues = queueFamilyIndices[0] == queueFamilyIndices[1];
	  vk::SharingMode sharingMode = !sameQueues ? vk::SharingMode::eConcurrent : vk::SharingMode::eExclusive;
	  swapinfo.imageExtent = surfaceCaps.currentExtent;
	  swapinfo.surface = surface_;
	  swapinfo.minImageCount = surfaceCaps.minImageCount + 1U + TRIPLE_BUFFERED;		// min(1) + 1(double buffered) + 1(triple buffered)
	  swapinfo.imageFormat = swapchainImageFormat_;
	  swapinfo.imageColorSpace = swapchainColorSpace_;
	  swapinfo.imageExtent = surfaceCaps.currentExtent;
	  swapinfo.imageArrayLayers = 1;
	  swapinfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
	  swapinfo.imageSharingMode = sharingMode;
	  swapinfo.queueFamilyIndexCount = !sameQueues ? 2 : 0;
	  swapinfo.pQueueFamilyIndices = queueFamilyIndices.data();
	  swapinfo.preTransform = surfaceCaps.currentTransform;;
	  swapinfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
	  swapinfo.presentMode = presentMode;
	  swapinfo.clipped = 1;
	  swapinfo.oldSwapchain = vk::SwapchainKHR{};
	  swapchain_ = device.createSwapchainKHRUnique(swapinfo);

	  images_ = device.getSwapchainImagesKHR(*swapchain_);
	  for (auto &img : images_) {
		  vk::ImageViewCreateInfo ci{};
		  ci.image = img;
		  ci.viewType = vk::ImageViewType::e2D;
		  ci.format = swapchainImageFormat_;
		  ci.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
		  imageViews_.emplace_back(device.createImageView(ci));
	  }

	  {
		  vk::CommandPoolCreateInfo cpci{ vk::CommandPoolCreateFlagBits::eTransient, graphicsQueueFamilyIndex };
		  commandPool_[eCommandPools::TRANSIENT_POOL] = device.createCommandPoolUnique(cpci);
	  }

	  auto memprops = physicalDevice.getMemoryProperties();
	  depthStencilImage_ = vku::DepthStencilImage(device, memprops, width_, height_, *commandPool_[eCommandPools::TRANSIENT_POOL], device.getQueue(graphicsQueueFamilyIndex, 0));

	  // Build the renderpass using two attachments, colour and depth/stencil. (regular rendering pass)
	  {
		  vku::RenderpassMaker rpm;

		  // The only colour attachment.
		  rpm.attachmentBegin(swapchainImageFormat_);
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eClear);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eStore);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eColorAttachmentOptimal);

		  // The depth/stencil attachment.
		  rpm.attachmentBegin(depthStencilImage_.format());
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eClear);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eStore);
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

		  // A subpass to render using the above two attachments.
		  rpm.subpassBegin(vk::PipelineBindPoint::eGraphics);
		  rpm.subpassColorAttachment(vk::ImageLayout::eColorAttachmentOptimal, 0);
		  rpm.subpassDepthStencilAttachment(vk::ImageLayout::eDepthStencilAttachmentOptimal, 1);

		  // A dependency to reset the layout of both attachments.
		  rpm.dependencyBegin(VK_SUBPASS_EXTERNAL, 0);
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite);

		  // Use the maker object to construct the vulkan object
		  renderPass_ = rpm.createUnique(device);
	  }

	  // Build the renderpass using one attachment, colour   (overlay / transparency pass)
	  {
		  vku::RenderpassMaker rpm;

		  // The colour attachment.
		  rpm.attachmentBegin(swapchainImageFormat_);
		  // Don't clear the framebuffer for overlay on top of main renderpass
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eLoad);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eStore);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eColorAttachmentOptimal);
		  rpm.attachmentFinalLayout(vk::ImageLayout::ePresentSrcKHR);

		  // A subpass to render using the above two attachments.
		  rpm.subpassBegin(vk::PipelineBindPoint::eGraphics);
		  rpm.subpassColorAttachment(vk::ImageLayout::eColorAttachmentOptimal, 0);

		  // 2 dependency to reset the layout of both attachments.
		  rpm.dependencyBegin(VK_SUBPASS_EXTERNAL, 0);
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eBottomOfPipe);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eMemoryRead);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  rpm.dependencyBegin(0, VK_SUBPASS_EXTERNAL);
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eBottomOfPipe);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eMemoryRead);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  // Use the maker object to construct the vulkan object
		  overlayPass_ = rpm.createUnique(device);
	  }

	  for (int i = 0; i != imageViews_.size(); ++i) {
		  vk::ImageView attachments[2] = { imageViews_[i], depthStencilImage_.imageView() };
		  vk::FramebufferCreateInfo fbci{ {}, *renderPass_, 2, attachments, width_, height_, 1 };
		  framebuffers_[eFrameBuffers::COLOR_DEPTH].push_back(device.createFramebufferUnique(fbci));
	  }
	  for (int i = 0; i != imageViews_.size(); ++i) {
		  vk::ImageView attachments[1] = { imageViews_[i] };
		  vk::FramebufferCreateInfo fbci{ {}, *overlayPass_, 1, attachments, width_, height_, 1 };
		  framebuffers_[eFrameBuffers::COLOR_ONLY].push_back(device.createFramebufferUnique(fbci));
	  }

	  {
		  vk::SemaphoreCreateInfo sci;
		  imageAcquireSemaphore_ = device.createSemaphoreUnique(sci);
		  commandCompleteSemaphore_ = device.createSemaphoreUnique(sci);
		  transferCompleteSemaphore_[0] = device.createSemaphoreUnique(sci); transferCompleteSemaphore_[1] = device.createSemaphoreUnique(sci);
		  computeSemaphore_ = device.createSemaphoreUnique(sci);
	  }
	  
	  {
		  vk::EventCreateInfo eci;
		  for (int i = 0; i != eGraphicsEvent::_size(); ++i) {
			  graphicsEvents[i] = device.createEventUnique(eci);
		  }
	  }
	  
	  typedef vk::CommandPoolCreateFlagBits ccbits;

	  {
		  vk::CommandPoolCreateInfo cpci{ ccbits::eTransient | ccbits::eResetCommandBuffer, graphicsQueueFamilyIndex };
		  commandPool_[eCommandPools::DEFAULT_POOL] = device.createCommandPoolUnique(cpci);
	  }
	  {
		  vk::CommandPoolCreateInfo cpci{ ccbits::eTransient | ccbits::eResetCommandBuffer, graphicsQueueFamilyIndex };
		  commandPool_[eCommandPools::OVERLAY_POOL] = device.createCommandPoolUnique(cpci);
	  }
	  {
		  vk::CommandPoolCreateInfo cpci{ ccbits::eTransient | ccbits::eResetCommandBuffer, graphicsQueueFamilyIndex };
		  commandPool_[eCommandPools::SCATTERED_POOL] = device.createCommandPoolUnique(cpci);
	  }
	  {

		  vk::CommandPoolCreateInfo cpci{ ccbits::eTransient | ccbits::eResetCommandBuffer, transferQueueFamilyIndex };
		  commandPool_[eCommandPools::DMA_TRANSFER_POOL_THREAD_PRIMARY] = device.createCommandPoolUnique(cpci);
		  commandPool_[eCommandPools::DMA_TRANSFER_POOL_THREAD_SECONDARY] = device.createCommandPoolUnique(cpci);
	  }
	  {
		  vk::CommandPoolCreateInfo cpci{ ccbits::eTransient | ccbits::eResetCommandBuffer, computeQueueFamilyIndex };
		  commandPool_[eCommandPools::COMPUTE_POOL] = device.createCommandPoolUnique(cpci);
	  }

	  // Create draw buffers
	  {
		  vk::CommandBufferAllocateInfo cbai{ *commandPool_[eCommandPools::DEFAULT_POOL], vk::CommandBufferLevel::ePrimary, (uint32_t)framebuffers_[eFrameBuffers::COLOR_DEPTH].size() };
		  staticDrawBuffers_.allocate(device, cbai);
		 
		  for (int i = 0; i != staticDrawBuffers_.size(); ++i) {
			  staticCommandsDirty_.emplace_back(false);
		  }
	  }
	  {
		  vk::CommandBufferAllocateInfo cbai{ *commandPool_[eCommandPools::OVERLAY_POOL], vk::CommandBufferLevel::ePrimary, (uint32_t)framebuffers_[eFrameBuffers::COLOR_ONLY].size() };
		  overlayDrawBuffers_.allocate<eOverlayBuffers::RENDER>(device, cbai);
	  }
	  {
		  vk::CommandBufferAllocateInfo cbai{ *commandPool_[eCommandPools::SCATTERED_POOL], vk::CommandBufferLevel::ePrimary, 1U };
		  scatteredDrawBuffer.allocate(device, cbai);
		  scattered_staticCommandsDirty_ = false;
	  }

	  {
		  {
			  vk::CommandBufferAllocateInfo cbai{ *commandPool_[eCommandPools::DMA_TRANSFER_POOL_THREAD_PRIMARY], vk::CommandBufferLevel::ePrimary, (uint32_t)framebuffers_[eFrameBuffers::COLOR_DEPTH].size() };
			  dynamicDrawBuffers_.allocate(device, cbai);
		  }
		  {

			  vk::CommandBufferAllocateInfo cbai{ *commandPool_[eCommandPools::DMA_TRANSFER_POOL_THREAD_PRIMARY], vk::CommandBufferLevel::ePrimary, 1U };
			  computeDrawBuffers_.allocate<eComputeBuffers::TRANSFER>(device, cbai);
		  }
		  {
			  vk::CommandBufferAllocateInfo cbai{ *commandPool_[eCommandPools::DMA_TRANSFER_POOL_THREAD_SECONDARY], vk::CommandBufferLevel::ePrimary, (uint32_t)framebuffers_[eFrameBuffers::COLOR_DEPTH].size() };
			  overlayDrawBuffers_.allocate<eOverlayBuffers::TRANSFER>(device, cbai);
		  }
	  }
	  {
		  vk::CommandBufferAllocateInfo cbai{ *commandPool_[eCommandPools::COMPUTE_POOL], vk::CommandBufferLevel::ePrimary, 1U };
		  computeDrawBuffers_.allocate<eComputeBuffers::COMPUTE>(device, cbai);
		  computeCommandsDirty_ = false;
	  }

    ok_ = true;
  }

  /// Dump the capabilities of the physical device used by this window.
  void dumpCaps(std::ostream &os, vk::PhysicalDevice pd) const {
    os << "Surface formats\n";
    auto fmts = pd.getSurfaceFormatsKHR(surface_);
    for (auto &fmt : fmts) {
      auto fmtstr = vk::to_string(fmt.format);
      auto cstr = vk::to_string(fmt.colorSpace);
      os << "format=" << fmtstr << " colorSpace=" << cstr << "\n";
    }

    os << "Present Modes\n";
    auto presentModes = pd.getSurfacePresentModesKHR(surface_);
    for (auto pm : presentModes) {
      std::cout << vk::to_string(pm) << "\n";
    }
  }

  static void defaultRenderFunc(vk::CommandBuffer cb, int imageIndex, vk::RenderPassBeginInfo const&rpbi) {
    vk::CommandBufferBeginInfo bi{};
    cb.begin(bi);
    cb.end();
  }

  static_renderpass_function_unconst staticCommandCache;
  scattered_offscreen_renderpass_function_unconst scatteredCommandCache;

  /// Build a static draw buffer. This will be rendered after any dynamic content generated in draw()
  void setStaticCommands(static_renderpass_function static_function, int32_t const iImageIndex = -1) {

	  if (iImageIndex < 0) {
		  for (int i = 0; i != staticDrawBuffers_.size(); ++i) {
			  vk::CommandBuffer cb = *staticDrawBuffers_.cb[0][i];

			  vk::ClearValue const clearArray[] = { vk::ClearValue{ std::array<uint32_t, 4>{0, 0, 0, 0}}, vk::ClearDepthStencilValue{1.0f, 0} };

			  vk::RenderPassBeginInfo rpbi;
			  rpbi.renderPass = *renderPass_;
			  rpbi.framebuffer = *framebuffers_[eFrameBuffers::COLOR_DEPTH][i];
			  rpbi.renderArea = vk::Rect2D{ {0, 0}, {width_, height_} };
			  rpbi.clearValueCount = (uint32_t)_countof(clearArray);
			  rpbi.pClearValues = clearArray;

			  static_renderpass s = { cb, graphicsEvents[eGraphicsEvent::RENDERED_STATIC], rpbi };
			  static_function(s);

			  staticCommandsDirty_[i] = false;
		  }
		  
		  staticCommandCache = static_function;
	  }
	  else {
		  vk::CommandBuffer cb = *staticDrawBuffers_.cb[0][iImageIndex];

		  vk::ClearValue const clearArray[] = { vk::ClearValue{ std::array<uint32_t, 4>{0, 0, 0, 0}}, vk::ClearDepthStencilValue{1.0f, 0} };

		  vk::RenderPassBeginInfo rpbi;
		  rpbi.renderPass = *renderPass_;
		  rpbi.framebuffer = *framebuffers_[eFrameBuffers::COLOR_DEPTH][iImageIndex];
		  rpbi.renderArea = vk::Rect2D{ {0, 0}, {width_, height_} };
		  rpbi.clearValueCount = (uint32_t)_countof(clearArray);
		  rpbi.pClearValues = clearArray;

		  static_renderpass s = { cb, graphicsEvents[eGraphicsEvent::RENDERED_STATIC], rpbi };
		  static_function(s);

		  staticCommandsDirty_[iImageIndex] = false;
	  }
  }

  void setStaticCommands(scattered_offscreen_renderpass_function scatttered_static_function, bool const cacheCommand = true) {
	vk::CommandBuffer cb = *scatteredDrawBuffer.cb[0][0];
	scattered_offscreen_renderpass s = { cb };
	scatttered_static_function(s);

	scattered_staticCommandsDirty_ = false;

	if (cacheCommand) {
		scatteredCommandCache = scatttered_static_function;
	}
  }
  void setStaticCommandsDirty(static_renderpass_function static_function) {

	  if (static_function == staticCommandCache) {
		  for (int i = 0; i != staticCommandsDirty_.size(); ++i) {
			  staticCommandsDirty_[i] = true;
		  }
	  }
#ifndef NDEBUG
	  assert_print(static_function == staticCommandCache, "[FAIL] No static command cache match");
#endif
  }
  void setStaticCommandsDirty(scattered_offscreen_renderpass_function scatttered_static_function) {
#ifndef NDEBUG
	  assert_print(!scattered_staticCommandsDirty_, "[FAIL] scattered static command aleady dirty");
#endif
	  if (scatttered_static_function == scatteredCommandCache)
		scattered_staticCommandsDirty_ = true;
#ifndef NDEBUG
	  assert_print(scatttered_static_function == scatteredCommandCache, "[FAIL] No scattered static command cache match");
#endif
  }
  /// Queue the static command buffer for the next image in the swap chain. Optionally call a function to create a dynamic command buffer
  /// for uploading textures, changing uniforms etc.
  void draw(const vk::Device& __restrict device, const vk::Queue& __restrict graphicsQueue, const vk::Queue& __restrict computeQueue,
	  compute_function gpu_compute, dynamic_renderpass_function dynamic_function, overlay_renderpass_function overlay_function, bool const bRenderScatterredPass = false) {

	  constexpr auto const umax = std::numeric_limits<uint64_t>::max();

	  uint32_t imageIndex = 0;
	  device.acquireNextImageKHR(*swapchain_, umax, *imageAcquireSemaphore_, vk::Fence(), &imageIndex);

	  // order of elements is important here, keeping eFences Layout/order
	  vk::CommandBuffer do_cb[2] = { *dynamicDrawBuffers_.cb[0][imageIndex], *overlayDrawBuffers_.cb[eOverlayBuffers::TRANSFER][imageIndex] };
	  vk::CommandBuffer ob = *overlayDrawBuffers_.cb[eOverlayBuffers::RENDER][imageIndex];

	  vk::Fence const& dynamic_fence = dynamicDrawBuffers_.fence[0][imageIndex];
	  vk::Fence const& overlay_fence = overlayDrawBuffers_.fence[eOverlayBuffers::RENDER][imageIndex];
	  vk::Fence const& dma_transfer_fence = computeDrawBuffers_.fence[eComputeBuffers::TRANSFER][0];
	  vk::Fence const& compute_fence = computeDrawBuffers_.fence[eComputeBuffers::COMPUTE][0];

	  bool const IsDirty(computeCommandsDirty_);		// save dirty flag to know if compute should be queued this frame

    vk::CommandBuffer compute_upload[2] = { *computeDrawBuffers_.cb[eComputeBuffers::TRANSFER][0], nullptr };
	{ // ######### begin main render update
		dynamic_renderpass dynamic = { do_cb[0], !computeDrawBuffers_.queued[eComputeBuffers::TRANSFER][0] };	// avoid lamda heap
		overlay_renderpass overlay = { do_cb[1], ob, graphicsEvents, vk::RenderPassBeginInfo(*overlayPass_, *framebuffers_[eFrameBuffers::COLOR_ONLY][imageIndex], vk::Rect2D{ {0, 0}, {width_, height_} } )}; // avoid lamda heap
		
		device.waitForFences(dynamic_fence, VK_TRUE, umax);
		device.resetFences(dynamic_fence);

		tbb::parallel_invoke(
			[dynamic_function, &dynamic, &device, &dma_transfer_fence, &compute_fence, &compute_upload, &gpu_compute, this] {

				constexpr auto const umax = std::numeric_limits<uint64_t>::max();

				if (computeDrawBuffers_.queued[eComputeBuffers::TRANSFER][0]) {
					device.waitForFences(dma_transfer_fence, VK_TRUE, umax);
					computeDrawBuffers_.queued[eComputeBuffers::TRANSFER][0] = false; // reset
				}
				device.resetFences(dma_transfer_fence);

				dynamic_function(dynamic);

				if (computeDrawBuffers_.queued[eComputeBuffers::COMPUTE][0]) {
					device.waitForFences(compute_fence, VK_TRUE, umax);
					computeDrawBuffers_.queued[eComputeBuffers::COMPUTE][0] = false; // reset
				}
				device.resetFences(compute_fence);

				compute_gpu_function compute = { compute_upload[eComputeBuffers::TRANSFER], compute_upload[eComputeBuffers::COMPUTE], computeCommandsDirty_ };

				computeCommandsDirty_ |= gpu_compute(compute);
			},
			[overlay_function, &overlay, &device, &overlay_fence] { 

				constexpr auto const umax = std::numeric_limits<uint64_t>::max();

				device.waitForFences(overlay_fence, VK_TRUE, umax);		// overlay fence is overlay Render cb
				device.resetFences(overlay_fence);
				overlay_function(overlay); 
			}
		);
	}
	
	vk::SubmitInfo submit{};

	vk::Semaphore tcSema[2] = { *transferCompleteSemaphore_[0], *transferCompleteSemaphore_[1] };
	vk::Semaphore ccSema = *commandCompleteSemaphore_;
	vk::Semaphore iaSema = *imageAcquireSemaphore_;
	
	// DMA TRANSFERS....
	
	// COMPUTE DMA TRANSFER SUBMIT //
	if (IsDirty) {
		vk::PipelineStageFlags waitStages{ vk::PipelineStageFlagBits::eTransfer };

		submit.waitSemaphoreCount = 1;
		submit.pWaitSemaphores = &iaSema;			// waiting for khr acquire
		submit.pWaitDstStageMask = &waitStages;
		submit.commandBufferCount = 1;				// submitting dma cb
		submit.pCommandBuffers = &compute_upload[eComputeBuffers::TRANSFER];
		submit.signalSemaphoreCount = 1;
		submit.pSignalSemaphores = &tcSema[0];			// signal for compute
		presentQueue().submit(1, &submit, dma_transfer_fence);

		computeDrawBuffers_.queued[eComputeBuffers::TRANSFER][0] = true;
	}
	// DYNAMIC & OVERLAY DYNAMIC SUBMIT //
	{
		vk::PipelineStageFlags waitStages{ vk::PipelineStageFlagBits::eTransfer };

		submit.waitSemaphoreCount = (uint32_t)!IsDirty;
		submit.pWaitSemaphores = &iaSema;			// waiting for khr acquire
		submit.pWaitDstStageMask = &waitStages;
		submit.commandBufferCount = 2;				// submitting dynamic cb & overlay's dynamic cb
		submit.pCommandBuffers = do_cb;
		submit.signalSemaphoreCount = 1;
		submit.pSignalSemaphores = &tcSema[1];			// signal for dynamic cb in slot 0, signal for overlay dynamic cb in slot 1 (completion)
		presentQueue().submit(1, &submit, dynamic_fence);
	}

	// COMPUTE SUBMIT *** first ***//
	bool bComputeActive = false;
	vk::Semaphore comSema = *computeSemaphore_;
	{
		if (IsDirty) {

			vk::CommandBuffer compute_process[2] = { nullptr, *computeDrawBuffers_.cb[eComputeBuffers::COMPUTE][0] };
			compute_gpu_function compute = { compute_process[eComputeBuffers::TRANSFER], compute_process[eComputeBuffers::COMPUTE], computeCommandsDirty_ };

			computeCommandsDirty_ = gpu_compute(compute);   // dirty flag is cleared if compute command buffer updates
															// OR remains false if already false

			vk::PipelineStageFlags waitStages{ vk::PipelineStageFlagBits::eComputeShader };

			submit.waitSemaphoreCount = 1;
			submit.pWaitSemaphores = &tcSema[0];				// waiting on transfer completion
			submit.pWaitDstStageMask = &waitStages;
			submit.commandBufferCount = 1;
			submit.pCommandBuffers = &compute_process[eComputeBuffers::COMPUTE];				// submitting compute cb
			submit.signalSemaphoreCount = 1;
			submit.pSignalSemaphores = &comSema;			// signalling compute cb completion
			computeQueue.submit(1, &submit, compute_fence);

			computeDrawBuffers_.queued[eComputeBuffers::COMPUTE][0] = true;
			bComputeActive = true;
		}
	}
	// STATIC SUBMIT //
	{
		vk::Fence const& static_fence = staticDrawBuffers_.fence[0][imageIndex];

		device.waitForFences(static_fence, VK_TRUE, umax);
		device.resetFences(static_fence);
		if (staticCommandsDirty_[imageIndex]) {
			setStaticCommands(staticCommandCache, imageIndex);
		}

		vk::CommandBuffer cb = *staticDrawBuffers_.cb[0][imageIndex];
		vk::PipelineStageFlags waitStages[2] = { vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTopOfPipe };
		vk::Semaphore tc_dynamic_compute[2] = { tcSema[1], comSema };

		if (bComputeActive)
		{
			submit.waitSemaphoreCount = 2;
			submit.pWaitSemaphores = tc_dynamic_compute;		// waiting on compute
			submit.pWaitDstStageMask = waitStages;
			submit.commandBufferCount = 1;
			submit.pCommandBuffers = &cb;				// submitting static cb
			submit.signalSemaphoreCount = 0;
			submit.pSignalSemaphores = nullptr;			// signalling static cb completion
		}
		else {
			submit.waitSemaphoreCount = 1;
			submit.pWaitSemaphores = &tcSema[1];		// waiting on transfer completion
			submit.pWaitDstStageMask = waitStages;
			submit.commandBufferCount = 1;
			submit.pCommandBuffers = &cb;				// submitting static cb
			submit.signalSemaphoreCount = 0;
			submit.pSignalSemaphores = nullptr;			// signalling static cb completion
		}
		
		graphicsQueue.submit(1, &submit, static_fence);
	}

	// SCATTERRED SUBMIT //
	if (bRenderScatterredPass) {	// only on scattered rendering enabled (transient)
		
		vk::Fence cbFenceScattered = scatteredDrawBuffer.fence[0][0];
		device.waitForFences(cbFenceScattered, VK_TRUE, umax);
		device.resetFences(cbFenceScattered);				// have to wait on associatted fence, and reset for next iteration

		if (scattered_staticCommandsDirty_) {
			setStaticCommands(this->scatteredCommandCache, false);
		}
															
		submit.waitSemaphoreCount = 0;						// not waiting on anything, dynamic cb is leveraged for scattered
		submit.pWaitSemaphores = nullptr;					// and the dynamic cb completion was already wait on by previous submit (static cb waiting on dynamic cb completion, signal already cleared)
		submit.pWaitDstStageMask = nullptr;
		submit.commandBufferCount = 1;
		submit.pCommandBuffers = &(*scatteredDrawBuffer.cb[0][0]); // submitting scattered's static cb
		submit.signalSemaphoreCount = 0;
		submit.pSignalSemaphores = nullptr;					// don't care when it completes, scattered's fence provides (parallel operation)
		graphicsQueue.submit(1, &submit, cbFenceScattered); // block here just in case. Scattered rendering only happens at interval of
															// time greater than time to complete the scattered dynamic + static time
															// also singular, not per swapchain image, so at least 1 iteration of all images in swapchain span must be done
	}														// before next invocation of scattered rendering (externally controlled from this function)

	// OVERLAY STATIC SUBMIT //
	{
		submit.waitSemaphoreCount = 0;				// waiting on overlay's dynamic cb (slot 0) completion and static cb (slot 1) completion
		submit.pWaitSemaphores = nullptr;			// prior submit already waited on &tcSema[1] (contains semaphor that represents dynamic + overlay transfer)
		submit.pWaitDstStageMask = nullptr;
		submit.commandBufferCount = 1;
		submit.pCommandBuffers = &ob;				// submitting overlay's static cb
		submit.signalSemaphoreCount = 1;
		submit.pSignalSemaphores = &ccSema;			// signalling commands complete
		graphicsQueue.submit(1, &submit, overlay_fence);	// ***static cb's associatted fence goes here with waiting on associatted semaphor
	}

	// PRESENT //
    vk::PresentInfoKHR presentInfo;
    vk::SwapchainKHR swapchain = *swapchain_;
    presentInfo.pSwapchains = &swapchain;
    presentInfo.swapchainCount = 1;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &ccSema;		// waiting on completion of overlay's static cb
    presentQueue().presentKHR(presentInfo);		// submit/present to screen queue
  }

  /// Return the queue family index used to present the surface to the display.
  uint32_t presentQueueFamily() const { return presentQueueFamily_; }

  /// Get the queue used to submit graphics jobs
  const vk::Queue presentQueue() const { return device_.getQueue(presentQueueFamily_, 0); }

  /// Return true if this window was created sucessfully.
  bool ok() const { return ok_; }

  /// Return the renderpass used by this window.
  vk::RenderPass const& __restrict renderPass() const { return(*renderPass_); }
  vk::RenderPass const& __restrict overlayPass() const { return(*overlayPass_); }

  /// Return the frame buffers used by this window
  const std::vector<vk::UniqueFramebuffer> &framebuffers(eFrameBuffers const index) const { return framebuffers_[index]; }

  size_t const framebufferCount(eFrameBuffers const index) const { return(framebuffers_[index].size()); }

  /// Destroy resources when shutting down.
  ~Window() {
    for (auto &iv : imageViews_) {
      device_.destroyImageView(iv);
    }
	
	computeDrawBuffers_.release(device_);
	staticDrawBuffers_.release(device_);
	dynamicDrawBuffers_.release(device_);
	overlayDrawBuffers_.release(device_);
	scatteredDrawBuffer.release(device_);

    swapchain_ = vk::UniqueSwapchainKHR{};
  }

  Window &operator=(Window &&rhs) = default;

  /// Return the width of the display.
  uint32_t width() const { return width_; }

  /// Return the height of the display.
  uint32_t height() const { return height_; }

  // return image views //
  vk::ImageView const depthstencilbufferview() const { return(depthStencilImage_.imageView()); }

  /// Return the format of the back buffer.
  vk::Format depthstencilImageFormat() const { return depthStencilImage_.format(); }

  /// Return the format of the back buffer.
  vk::Format swapchainImageFormat() const { return swapchainImageFormat_; }

  /// Return the colour space of the back buffer (Usually sRGB)
  vk::ColorSpaceKHR swapchainColorSpace() const { return swapchainColorSpace_; }

  /// Return the swapchain object
  const vk::SwapchainKHR swapchain() const { return *swapchain_; }

  /// Return the views of the swap chain images
  const std::vector<vk::ImageView> &imageViews() const { return imageViews_; }

  /// Return the swap chain images
  const std::vector<vk::Image> &images() const { return images_; }

  /// Return the semaphore signalled when an image is acquired.
  vk::Semaphore imageAcquireSemaphore() const { return *imageAcquireSemaphore_; }

  /// Return the semaphore signalled when the command buffers are finished.
  vk::Semaphore commandCompleteSemaphore() const { return *commandCompleteSemaphore_; }

  vk::Fence const& scatteredRenderingFence() const { return(scatteredDrawBuffer.fence[0][0]); }

  /// Return a defult command Pool to use to create new command buffers.
  vk::CommandPool const& commandPool(eCommandPools const index) const { return(*commandPool_[index]); }

  /// Return the number of swap chain images.
  int numImageIndices() const { return (int)images_.size(); }

private:
  vk::Instance instance_;
  vk::SurfaceKHR surface_;
  vk::UniqueSwapchainKHR swapchain_;
  vk::UniqueRenderPass renderPass_, overlayPass_;
  
  vk::UniqueSemaphore imageAcquireSemaphore_;
  vk::UniqueSemaphore commandCompleteSemaphore_;
  vk::UniqueSemaphore transferCompleteSemaphore_[2];
  vk::UniqueSemaphore computeSemaphore_;
 
  vk::UniqueEvent	  graphicsEvents[eGraphicsEvent::_size()];
  
  vk::UniqueCommandPool		commandPool_[eCommandPools::_size()];

  std::vector<vk::ImageView> imageViews_;
  std::vector<vk::Image> images_;
  
  std::vector<vk::UniqueFramebuffer> framebuffers_[2];
  CommandBufferContainer<2> computeDrawBuffers_;
  CommandBufferContainer<1> staticDrawBuffers_;
  CommandBufferContainer<1> dynamicDrawBuffers_;
  CommandBufferContainer<2> overlayDrawBuffers_;
  CommandBufferContainer<1> scatteredDrawBuffer;

  vku::DepthStencilImage depthStencilImage_;

  uint32_t presentQueueFamily_ = 0;
  uint32_t width_;
  uint32_t height_;
  vk::Format swapchainImageFormat_ = vk::Format::eB8G8R8A8Unorm;
  vk::ColorSpaceKHR swapchainColorSpace_ = vk::ColorSpaceKHR::eSrgbNonlinear;
  vk::Device device_;
  bool ok_ = false;

  std::vector<bool> staticCommandsDirty_;
  bool computeCommandsDirty_ = false;
  bool scattered_staticCommandsDirty_ = false;
};

} // namespace vku

#endif // VKU_FRAMEWORK_HPP
