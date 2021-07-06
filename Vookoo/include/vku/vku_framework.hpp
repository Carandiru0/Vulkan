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

#include <vku/vku.hpp>		// must place here and ONLY here

#ifndef VKU_NO_GLFW
#define GLFW_EXPOSE_NATIVE_WIN32
//#define GLFW_INCLUDE_VULKAN // not required here, as vulkan.h is already included
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
#include <unordered_map>
#include <vector>
#include <chrono>
#include <functional>
#include <cstddef>
#include <set>

#ifdef VKU_IMPLEMENTATION
#include <queue>
#include <Utility/async_long_task.h>

#endif
namespace vku {

	static inline constexpr vk::SampleCountFlagBits const DefaultSampleCount(vk::SampleCountFlagBits::e4); // 4xMSAA guarenteed supported, higher not really needed and can be overriden in driver control panel (radeon/nvidia tweaking) by user if wanted to enhance at a great loss in performance

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

	  uint32_t const apiVersion(VULKAN_API_VERSION_USED);

    vku::InstanceMaker im{};
    im.defaultLayers();
	im.applicationName(name.c_str());
	im.engineName("supersinfulsilicon");
	im.applicationVersion(1);
	im.engineVersion(1);
	im.apiVersion(apiVersion);

	// add additonal extensions
	auto const eps = vk::enumerateInstanceExtensionProperties().value;
	for (auto const& i : eps)
	{
		// if instance extension is available, add it. 

		if (bHDRExtensionEnabled) {
			// VK_EXT_swapchain_colorspace
			if (std::string_view(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME) == std::string_view(i.extensionName)) {
				im.extension(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
				bExtendedColorspaceOn = true;
			}
		}
	}

	// finally, create the actual instance //
    instance_ = im.createUnique();

#ifndef NDEBUG
    callback_ = DebugCallback(*instance_);
#endif

    auto const pds = instance_->enumeratePhysicalDevices().value;
	for (auto const& i : pds)
	{
		uint32_t const physicalDeviceApiVersion = i.getProperties().apiVersion;
		if (physicalDeviceApiVersion >= apiVersion) {
			physical_device_ = i;
			fmt::print(fg(fmt::color::magenta),  "[ Vulkan {:d}.{:d} ]" "\n", VK_VERSION_MAJOR(apiVersion),
																		      VK_VERSION_MINOR(apiVersion));
			fmt::print(fg(fmt::color::white),   "[ {:s} ]" "\n", (std::string_view const)i.getProperties().deviceName);
			break;
		}
	}
	if (!physical_device_) {
		fmt::print(fg(fmt::color::red),   "[ ! Vulkan {:d}.{:d} - Not supported by any gpu device ! ]" "\n", VK_VERSION_MAJOR(apiVersion),
																											 VK_VERSION_MINOR(apiVersion));
		for (auto const& i : pds)
		{
			fmt::print(fg(fmt::color::red), "[ {:s} ]" "\n", (std::string_view const)i.getProperties().deviceName);
		}
		return;
	}
    auto qprops = physical_device_.getQueueFamilyProperties();
    
    graphicsQueueFamilyIndex_ = 0;
    computeQueueFamilyIndex_ = 0;
	transferQueueFamilyIndex_ = 0;

	vk::QueueFlags
		searchGraphics = vk::QueueFlagBits::eGraphics,
		searchCompute = vk::QueueFlagBits::eCompute;  // **************** any ShaderReadOnlyOptimal (not general)sampler access in compute shader requires graphics queue aswell. Found out also that for compute to not stall the raphics pieline, it must be on a seperate queue, then it is correctly async compute
		 // speedy 8x8 granularity (multiple of 8) transfer queue is searched for differently (see below)
		 // ** resolution must be divisible by 8 (all normally are)

    // Look for an omnipurpose queue family first
    // It is better if we can schedule operations without barriers and semaphores.
    // The Spec says: "If an implementation exposes any queue family that supports graphics operations,
    // at least one queue family of at least one physical device exposed by the implementation
    // must support both graphics and compute operations."
    // Also: All commands that are allowed on a queue that supports transfer operations are
    // also allowed on a queue that supports either graphics or compute operations...
    // As a result we can expect a queue family with at least all three and maybe all four modes.

	uint32_t lastGranularity(0);

    for (int32_t qi = (int32_t)qprops.size() - 1; qi >= 0; --qi) {	// start from back to capture unique queues first
      auto &qprop = qprops[qi];

      if (searchGraphics && (qprop.queueFlags & searchGraphics) == searchGraphics) {
			graphicsQueueFamilyIndex_ = qi;
			if (0 == qi) {
				searchGraphics = (vk::QueueFlagBits)0; // prevent further search only if equal to zero for graphics queue (default index)
			}
			FMT_LOG_OK(GPU_LOG, "Graphics Queue Selected < {:s} {:d} >", vk::to_string(qprop.queueFlags), graphicsQueueFamilyIndex_);
      }
	  if (searchCompute && ((qprop.queueFlags & searchCompute) == searchCompute) && qprop.queueCount >= 2) { // also ensure there is 2 available compute queues
		  computeQueueFamilyIndex_ = qi;
		  searchCompute = (vk::QueueFlagBits)0; // prevent further search
		  FMT_LOG_OK(GPU_LOG, "Compute Queue Selected < {:s} {:d} >", vk::to_string(qprop.queueFlags), computeQueueFamilyIndex_);
	  }
	  
	  if ((qprop.queueFlags & vk::QueueFlagBits::eTransfer) == vk::QueueFlagBits::eTransfer &&
		!((qprop.queueFlags & vk::QueueFlagBits::eCompute) == vk::QueueFlagBits::eCompute)) { // finding dedicated transfer queue (does not have compute capability)
																							  // checked hw database, compatible with minimum spec of Radeon 290 and Nvidia GTX 970
																							  // will have at least 2 queues for transfer
		  uint32_t const granularity(qprop.minImageTransferGranularity.width + qprop.minImageTransferGranularity.height + qprop.minImageTransferGranularity.depth);

		  if (granularity > lastGranularity) {

			  // supporting only queues with:
			  //
			  // qprop.minImageTransferGranularity.width	= 1   or is divisable by 8
			  // qprop.minImageTransferGranularity.height	= 1   or ""  ""   ""  by 8
			  // qprop.minImageTransferGranularity.depth	= 1   or ""  ""   ""  by 8

			  if ((3 == granularity) || (0 == (granularity % 8))) {
				  transferQueueFamilyIndex_ = qi;
				  lastGranularity = granularity;
			  }
		  }
	  }
    }
	// error out if there is *not* 2 compute queues or no compute queue at all
	if (!searchCompute) { // search found compute queue

		if (!(qprops[computeQueueFamilyIndex_].queueCount >= 2)) {
			FMT_LOG_FAIL(GPU_LOG, "Simultaneous Compute Queues not supported!!");
			return;
		}
	}
	else {
		FMT_LOG_FAIL(GPU_LOG, "No Compute Queue found!!");
		return;
	}
	// error out if there is *not* 2 transfer queues
	if (!(qprops[transferQueueFamilyIndex_].queueCount >= 2)) {
		FMT_LOG_FAIL(GPU_LOG, "Simultaneous Transfer Queues not supported!!");
		return;
	}

	// Transfer quewe is selected by granularity. eg.) 8x8x8 min granularity allows for faster clears, uploads via dma if image has dimensions divisable by 8, etc.
	FMT_LOG_OK(GPU_LOG, "Transfer Queue Selected < {:s} {:d} >", vk::to_string(qprops[transferQueueFamilyIndex_].queueFlags), transferQueueFamilyIndex_);

	vk::PhysicalDeviceVulkanMemoryModelFeatures supportedMemoryModel{};
	vk::PhysicalDevice8BitStorageFeatures supportedByteStorage{};

	vk::PhysicalDeviceFeatures2 supportedFeatures{};

	// ################ start of pNext linked list chain for query
	supportedByteStorage.pNext = nullptr;
	supportedMemoryModel.pNext = &supportedByteStorage;
	supportedFeatures.pNext = &supportedMemoryModel;  // ####### pNext chain ###### for query support of extensions //


	physical_device_.getFeatures2(&supportedFeatures);

	vk::PhysicalDeviceFeatures enabledFeatures{};

	enabledFeatures.geometryShader = supportedFeatures.features.geometryShader;
	enabledFeatures.samplerAnisotropy = supportedFeatures.features.samplerAnisotropy;
	//enabledFeatures.robustBufferAccess = supportedFeatures.robustBufferAccess; // safer but a lot slower good for debugging out of bounds access
	enabledFeatures.textureCompressionBC = supportedFeatures.features.textureCompressionBC;
	enabledFeatures.independentBlend = supportedFeatures.features.independentBlend;
	enabledFeatures.shaderStorageImageExtendedFormats = supportedFeatures.features.shaderStorageImageExtendedFormats;
	enabledFeatures.vertexPipelineStoresAndAtomics = supportedFeatures.features.vertexPipelineStoresAndAtomics;
	enabledFeatures.fragmentStoresAndAtomics = supportedFeatures.features.fragmentStoresAndAtomics;

	PRINT_FEATURE(supportedMemoryModel.vulkanMemoryModel, "vulkan memory model"); if (!supportedMemoryModel.vulkanMemoryModel) return;
	PRINT_FEATURE(supportedByteStorage.storageBuffer8BitAccess, "storage buffer 8bit"); if (!supportedByteStorage.storageBuffer8BitAccess) return;
	PRINT_FEATURE(enabledFeatures.geometryShader, "geometry shader"); if (!enabledFeatures.geometryShader) return;
	PRINT_FEATURE(enabledFeatures.samplerAnisotropy, "anisotropic filtering"); if (!enabledFeatures.samplerAnisotropy) return;
	PRINT_FEATURE(enabledFeatures.textureCompressionBC, "texture compression"); if (!enabledFeatures.textureCompressionBC) return;
	PRINT_FEATURE(enabledFeatures.independentBlend, "independent blending"); if (!enabledFeatures.independentBlend) return;	  // independent (different) blend states for multiple color attachments
	PRINT_FEATURE(enabledFeatures.shaderStorageImageExtendedFormats, "extended compute image formats"); if (!enabledFeatures.shaderStorageImageExtendedFormats) return;
	PRINT_FEATURE(enabledFeatures.vertexPipelineStoresAndAtomics, "vertex image ops"); if (!enabledFeatures.vertexPipelineStoresAndAtomics) return;	  // use of image operations in vertex shader requires this feature to be enabled
	PRINT_FEATURE(enabledFeatures.fragmentStoresAndAtomics, "fragment image ops"); if (!enabledFeatures.fragmentStoresAndAtomics) return;	  // use of image operations in vertex shader requires this feature to be enabled

    vku::DeviceMaker dm{};
    dm.defaultLayers();

	// add extensions
	bool supported(false), memorybudget(false);
	auto const extensions = physical_device_.enumerateDeviceExtensionProperties().value;

	// Required Extensions //
	// internally promoted in vulkan 1.2 ADD_EXTENSION(extensions, dm, VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME, supported); if (!supported) return;
	ADD_EXTENSION(extensions, dm, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME, memorybudget); // optional, can use internal tracking of memory in vma if not available
	// internally promoted in vulkan 1.1 ADD_EXTENSION(extensions, dm, VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME, supported); if (!supported) return;
	// internally promoted in vulkan 1.1 ADD_EXTENSION(extensions, dm, VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME, supported); if (!supported) return;
	ADD_EXTENSION(extensions, dm, VK_EXT_GLOBAL_PRIORITY_EXTENSION_NAME, supported); if (!supported) return;
	// internally promoted in vulkan 1.2 ADD_EXTENSION(extensions, dm, VK_KHR_8BIT_STORAGE_EXTENSION_NAME, supported); if (!supported) return;// The code:StorageBuffer8BitAccess capability must: be supported by all
																										 // implementations of this extension, so no need to query further details as only ssbo 8bit is used. Uniform buffer objects with 8bit is not used.
	
	// bugfix: theses extensions only allowe a difference between a depth multisample count and a color multisample count
	//         so a difference between a color and another color multisample count still cannot be different
	//         these extensions are not required
	//ADD_EXTENSION(extensions, dm, VK_AMD_MIXED_ATTACHMENT_SAMPLES_EXTENSION_NAME, supported);  // was for mouse picking
	//ADD_EXTENSION(extensions, dm, VK_NV_FRAMEBUFFER_MIXED_SAMPLES_EXTENSION_NAME, supported);  // was  ""   ""     ""

	// Optional/Additional Extensions & detailed configuration if required by extension//
#if defined(FULLSCREEN_EXCLUSIVE) && defined(VK_EXT_full_screen_exclusive)
	if (bFullScreenExclusiveExtensionEnabled) {
		ADD_EXTENSION(extensions, dm, VK_EXT_FULL_SCREEN_EXCLUSIVE_EXTENSION_NAME, supported);
		bFullScreenExclusiveExtensionSupported = supported;
	}
#endif

#if defined(VK_EXT_hdr_metadata)
	if (bHDRExtensionEnabled & bExtendedColorspaceOn) {
		ADD_EXTENSION(extensions, dm, VK_EXT_HDR_METADATA_EXTENSION_NAME, supported);
		bHDRExtensionSupported = supported;
	}
#endif

	// Create ****QUEUES**** //
    dm.queue<VK_QUEUE_GLOBAL_PRIORITY_REALTIME_EXT>(graphicsQueueFamilyIndex_);			// Add graphics queue as first queue for device
	dm.queue<VK_QUEUE_GLOBAL_PRIORITY_REALTIME_EXT>(computeQueueFamilyIndex_, 2);		// 2 compute queues (support minimum spec Radeon 290, Hvidia GTX 970)
	dm.queue<VK_QUEUE_GLOBAL_PRIORITY_REALTIME_EXT>(transferQueueFamilyIndex_, 2);		// Last, 2 transfer dma queues (supports minimum spec Radeon 290, Hvidia GTX 970)


	// ################ start of pNext linked list chain for device creation
	vk::PhysicalDevice8BitStorageFeatures byteStorage{
		VK_TRUE,												 // - required (supportedByteStorage.storageBuffer8BitAccess) //
		supportedByteStorage.uniformAndStorageBuffer8BitAccess,  // - optional //
		supportedByteStorage.storagePushConstant8				 // - optional //
	};
	byteStorage.pNext = nullptr;

	vk::PhysicalDeviceVulkanMemoryModelFeatures memoryModel{
		VK_TRUE,																 // - required (supportedMemoryModel.vulkanMemoryModel) //
		supportedMemoryModel.vulkanMemoryModelDeviceScope,						 // - optional //
		supportedMemoryModel.vulkanMemoryModelAvailabilityVisibilityChains		 // - optional //
	};
	memoryModel.pNext = &byteStorage; /// ######### pNext chain ############## (reference previous extension)


    device_ = dm.createUnique(physical_device_, enabledFeatures, &memoryModel);
    
    vk::PipelineCacheCreateInfo pipelineCacheInfo{};
    pipelineCache_ = device_->createPipelineCacheUnique(pipelineCacheInfo).value;

    std::vector<vk::DescriptorPoolSize> poolSizes;
    poolSizes.emplace_back(vk::DescriptorType::eUniformBuffer, MAX_NUM_UNIFORM_BUFFERS);
    poolSizes.emplace_back(vk::DescriptorType::eCombinedImageSampler, MAX_NUM_IMAGES);
    poolSizes.emplace_back(vk::DescriptorType::eStorageBuffer, MAX_NUM_STORAGE_BUFFERS);

    // Create an arbitrary number of descriptors in a pool.
    // Allow the descriptors to be freed, possibly not optimal behaviour.
    vk::DescriptorPoolCreateInfo descriptorPoolInfo{};
	//descriptorPoolInfo.flags = // vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet; // not recommended by AMD
    descriptorPoolInfo.maxSets = MAX_NUM_DESCRIPTOR_SETS;
    descriptorPoolInfo.poolSizeCount = (uint32_t)poolSizes.size();
    descriptorPoolInfo.pPoolSizes = poolSizes.data();
    descriptorPool_ = device_->createDescriptorPoolUnique(descriptorPoolInfo).value;

#ifndef NDEBUG
	callback_.acquireDeviceFunctionPointers(*device_);
#endif

	// create vma global singleton instance in vku framework
	VmaAllocatorCreateInfo allocatorInfo{};
	allocatorInfo.vulkanApiVersion = VULKAN_API_VERSION_USED;
	allocatorInfo.flags = VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT | (memorybudget ? VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT : (VmaAllocatorCreateFlags)0);
	allocatorInfo.instance = *instance_;
	allocatorInfo.physicalDevice = physical_device_;
	allocatorInfo.device = *device_;
	//allocatorInfo.pVulkanFunctions

	ok_ = (VkResult::VK_SUCCESS == vmaCreateAllocator(&allocatorInfo, &vma_));  // note, as requirement of application, allocator singleton can only be access by a single thread at any given time!

  }

  /// Get the Vulkan instance.
  const vk::Instance instance() const { return *instance_; }

  /// Get the Vulkan device.
  const vk::Device device() const { return *device_; }

  /// Get the physical device.
  const vk::PhysicalDevice &physicalDevice() const { return physical_device_; }

  /// Get the default pipeline cache (you can use your own if you like).
  const vk::PipelineCache pipelineCache() const { return *pipelineCache_; }

  /// Get the default descriptor pool (you can use your own if you like).
  const vk::DescriptorPool descriptorPool() const { return *descriptorPool_; }

  /// Get the family index for the graphics queues.
  uint32_t const graphicsQueueFamilyIndex() const { return graphicsQueueFamilyIndex_; }

  /// Get the family index for the compute queues.
  uint32_t const computeQueueFamilyIndex() const { return computeQueueFamilyIndex_; }

  /// Get the family index for the compute queues.
  uint32_t const transferQueueFamilyIndex() const { return transferQueueFamilyIndex_; }

  /// Clean up the framework satisfying the Vulkan verification layers.
  ~Framework() {

    if (device_) {
      device_->waitIdle();

	  if (vma_) {
		  vmaDestroyAllocator(vma_); vma_ = nullptr;
	  }

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

  // extensions supported ? //
  bool const isFullScreenExclusiveExtensionSupported() const {
	  return(bFullScreenExclusiveExtensionEnabled & bFullScreenExclusiveExtensionSupported);
  }
  void setFullScreenExclusiveEnabled(bool const bEnabled) {
	  bFullScreenExclusiveExtensionEnabled = bEnabled;
  }


  bool const isHDRExtensionSupported() const {
	  return(bHDRExtensionEnabled & bHDRExtensionSupported);
  }
  uint32_t const getMaximumNits() const {
	  return(max_hdr_monitor_nits);
  }
  void setHDREnabled(bool const bEnabled, uint32_t const max_nits) {

	  if (bEnabled && 0 != max_nits) // prevent zero max nits being a valid state while hdr is enabled.
	  {
		  bHDRExtensionEnabled = true;
		  max_hdr_monitor_nits = max_nits;
	  }
	  else {
		  bHDRExtensionEnabled = false;
		  max_hdr_monitor_nits = 0;
	  }
  }

  /// Returns true if the Framework has been built correctly.
  bool ok() const { return ok_; }

private:
  vk::UniqueInstance instance_;
#ifndef NDEBUG
  vku::DebugCallback callback_;
#endif
  vk::UniqueDevice device_;
  vk::PhysicalDevice physical_device_;
  vk::UniquePipelineCache pipelineCache_;
  vk::UniqueDescriptorPool descriptorPool_;
  uint32_t graphicsQueueFamilyIndex_;
  uint32_t computeQueueFamilyIndex_;
  uint32_t transferQueueFamilyIndex_;

  // extensions supported ? //
  bool bFullScreenExclusiveExtensionEnabled = false,
	   bFullScreenExclusiveExtensionSupported = false;

  uint32_t max_hdr_monitor_nits = 0u;
  bool bExtendedColorspaceOn = false,
	   bHDRExtensionEnabled = false,
	   bHDRExtensionSupported = false;

  bool ok_ = false;
};

/// This class wraps a window, a surface and a swap chain for that surface.
BETTER_ENUM(eCommandPools, uint32_t const, DEFAULT_POOL = 0, OVERLAY_POOL, TRANSIENT_POOL, DMA_TRANSFER_POOL_PRIMARY, DMA_TRANSFER_POOL_SECONDARY, COMPUTE_POOL_PRIMARY, COMPUTE_POOL_SECONDARY);
BETTER_ENUM(eFrameBuffers, uint32_t const, DEPTH, HALF_COLOR_ONLY, FULL_COLOR_ONLY, MID_COLOR_DEPTH, COLOR_DEPTH, PRESENT, OFFSCREEN);
BETTER_ENUM(eOverlayBuffers, uint32_t const, TRANSFER, RENDER);
BETTER_ENUM(eComputeBuffers, uint32_t const, TRANSFER_LIGHT, COMPUTE_LIGHT, COMPUTE_TEXTURE);

class Window {

	Framework const& fw_;	// reference to framework!

public:
	Window(vku::Framework const& fw ) : fw_(fw) {}

#ifndef VKU_NO_GLFW
  /// Construct a window, surface and swapchain using a GLFW window.
  Window(vku::Framework const & fw, const vk::Device &device, const vk::PhysicalDevice &physicalDevice, uint32_t const graphicsQueueFamilyIndex, uint32_t const computeQueueFamilyIndex, uint32_t const transferQueueFamilyIndex, GLFWwindow * const window, bool const bVsyncDisabled)
	  : fw_(fw)
  {
#ifdef VK_USE_PLATFORM_WIN32_KHR
    auto module_handle = GetModuleHandle(nullptr);
    auto const handle = glfwGetWin32Window(window);
	glfwSetWindowUserPointer(window, this);
    auto const ci = vk::Win32SurfaceCreateInfoKHR{{}, module_handle, handle};
    auto const surface = fw.instance().createWin32SurfaceKHR(ci).value;
	auto const monitor = MonitorFromWindow(handle, MONITOR_DEFAULTTOPRIMARY);
#endif
#ifdef VK_USE_PLATFORM_XLIB_KHR
    auto display = glfwGetX11Display();
    auto x11window = glfwGetX11Window(window);
    auto ci = vk::XlibSurfaceCreateInfoKHR{{}, display, x11window};
    auto surface = instance.createXlibSurfaceKHR(ci);
#endif
    init(fw.instance(), device, physicalDevice, graphicsQueueFamilyIndex, computeQueueFamilyIndex, transferQueueFamilyIndex, surface, monitor, bVsyncDisabled);
  }
#endif

 /* void downsampleDepth(vk::CommandBuffer const&__restrict cb)
  {
	  // layout transitions must be set appropriately b4 function

	  vk::ImageSubresourceLayers const srcLayer(vk::ImageAspectFlagBits::eDepth, 0, 0, 1);
	  vk::ImageSubresourceLayers const dstLayer(vk::ImageAspectFlagBits::eDepth, 0, 0, 1);

	  std::array<vk::Offset3D, 2> const srcOffsets = { vk::Offset3D(), vk::Offset3D(width_, height_, 1) };
	  std::array<vk::Offset3D, 2> const dstOffsets = { vk::Offset3D(), vk::Offset3D(width_ / vku::DOWN_RES_FACTOR, height_ / vku::DOWN_RES_FACTOR, 1) };

	  vk::ImageBlit const region(srcLayer, srcOffsets, dstLayer, dstOffsets);

	  cb.blitImage(depthImage_.image(), vk::ImageLayout::eTransferSrcOptimal, depthImageDown_.image(), vk::ImageLayout::eTransferDstOptimal, 1, &region, vk::Filter::eNearest);
  }
  */
  /*
  void copyLastRenderedImage(vk::CommandBuffer const& __restrict cb, vku::TextureImage2D* const __restrict dstImage, uint32_t const last_image_index)
  {
	  // layout transitions must be set appropriately b4 function for destination image
	  // swap chain image is managed inside this function only
	  vk::ImageMemoryBarrier imageMemoryBarriers = {};
	  imageMemoryBarriers.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	  imageMemoryBarriers.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	  imageMemoryBarriers.image = images_[last_image_index];
	  imageMemoryBarriers.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };

	  vk::PipelineStageFlags srcStageMask{};
	  vk::PipelineStageFlags dstStageMask{};
	  vk::DependencyFlags dependencyFlags{};
	  vk::AccessFlags srcMask{};
	  vk::AccessFlags dstMask{};

	  typedef vk::ImageLayout il;
	  typedef vk::PipelineStageFlagBits psfb;
	  typedef vk::AccessFlagBits afb;

	  imageMemoryBarriers.oldLayout = vk::ImageLayout::ePresentSrcKHR;
	  imageMemoryBarriers.newLayout = vk::ImageLayout::eTransferSrcOptimal;
	  srcMask = afb::eMemoryRead; srcStageMask = psfb::eBottomOfPipe;
	  dstMask = afb::eTransferRead; dstStageMask = psfb::eTransfer;

	  imageMemoryBarriers.srcAccessMask = srcMask;
	  imageMemoryBarriers.dstAccessMask = dstMask;
	  auto memoryBarriers = nullptr;
	  auto bufferMemoryBarriers = nullptr;
	  cb.pipelineBarrier(srcStageMask, dstStageMask, dependencyFlags, memoryBarriers, bufferMemoryBarriers, imageMemoryBarriers);


	  // do gpu -> gpu local blit
	  // must be a blit because we need it back in linear color space, swapchain images are srgb textures
	  // the blit performs the conversion back to linear!
	  {
		  vk::ImageSubresourceLayers const srcLayer(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
		  vk::ImageSubresourceLayers const dstLayer(vk::ImageAspectFlagBits::eColor, 0, 0, 1);

		  std::array<vk::Offset3D, 2> const srcOffsets = { vk::Offset3D(), vk::Offset3D(width_, height_, 1) };
		  std::array<vk::Offset3D, 2> const dstOffsets = { vk::Offset3D(), vk::Offset3D(width_, height_, 1) };

		  vk::ImageBlit const region(srcLayer, srcOffsets, dstLayer, dstOffsets);

		  cb.blitImage(images_[last_image_index], vk::ImageLayout::eTransferSrcOptimal, dstImage->image(), vk::ImageLayout::eTransferDstOptimal, 1, &region, vk::Filter::eNearest);
	  }


	  // automatically transition last rendered swapchain image back to swapchain friendly layout
	  imageMemoryBarriers.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
	  imageMemoryBarriers.newLayout = vk::ImageLayout::ePresentSrcKHR;
	  srcMask = afb::eTransferRead; srcStageMask = psfb::eTransfer;
	  dstMask = afb::eMemoryRead; dstStageMask = psfb::eBottomOfPipe;

	  imageMemoryBarriers.srcAccessMask = srcMask;
	  imageMemoryBarriers.dstAccessMask = dstMask;
	  cb.pipelineBarrier(srcStageMask, dstStageMask, dependencyFlags, memoryBarriers, bufferMemoryBarriers, imageMemoryBarriers);
  }*/

  private:

  bool const recreateSwapChain()
  {
	  fmt::print(fg(fmt::color::lime_green), "creating swapchain.... " "\n");

	  vk::SurfaceCapabilities2KHR surfaceCaps{};
	  vk::PhysicalDeviceSurfaceInfo2KHR surfaceInfo(surface_);

	  surfaceCaps.pNext = nullptr; // safe

#if defined(FULLSCREEN_EXCLUSIVE) && defined(VK_EXT_full_screen_exclusive)
	  vk::SurfaceFullScreenExclusiveWin32InfoEXT full_screen_exclusive_win32(monitor_);
	  vk::SurfaceFullScreenExclusiveInfoEXT full_screen_exclusive{ vk::FullScreenExclusiveEXT::eApplicationControlled };
	  vk::SurfaceCapabilitiesFullScreenExclusiveEXT surfaceCapFullscreenExclusive{};

	  if (nullptr == monitor_) {
		  fmt::print(fg(fmt::color::orange), "fullscreen exclusive disabled." "\n");
	  }
	  else {
		  
		  if (fw_.isFullScreenExclusiveExtensionSupported()) {
			  full_screen_exclusive.pNext = &full_screen_exclusive_win32;
			  surfaceInfo.pNext = &full_screen_exclusive;
			  surfaceCaps.pNext = &surfaceCapFullscreenExclusive;
		  }
	  }
#endif

	  physicalDevice_.getSurfaceCapabilities2KHR(&surfaceInfo, &surfaceCaps);
	  width_ = surfaceCaps.surfaceCapabilities.currentExtent.width;
	  height_ = surfaceCaps.surfaceCapabilities.currentExtent.height;

#if defined(FULLSCREEN_EXCLUSIVE) && defined(VK_EXT_full_screen_exclusive)
	  if (monitor_ && fw_.isFullScreenExclusiveExtensionSupported()) {
		  bFullScreenExclusiveOn = surfaceCapFullscreenExclusive.fullScreenExclusiveSupported;
		  if (bFullScreenExclusiveOn) {
			  fmt::print(fg(fmt::color::lime_green), "fullscreen exclusive\n");
		  }
	  }
	  else { // Extension doesn't exist or user settings ini has disabled exclusivity, silently fail/disable exclusive fullscreen
		  bFullScreenExclusiveOn = false;
	  }
#endif

	  auto fmts = physicalDevice_.getSurfaceFormats2KHR(surfaceInfo).value;
	  // default to first format
	  swapchainImageFormat_ = fmts[0].surfaceFormat.format;
	  swapchainColorSpace_ = fmts[0].surfaceFormat.colorSpace;

	  // returned errornouse result from driver? Default to preferred 8bit format.
	  if (fmts.size() == 1 && swapchainImageFormat_ == vk::Format::eUndefined) {
		  swapchainImageFormat_ = vk::Format::eB8G8R8A8Unorm;
		  swapchainColorSpace_ = vk::ColorSpaceKHR::eSrgbNonlinear;
	  }
	  else { // otherwise find optimal format

		  // search for 10bit HDR target
		  if (bFullScreenExclusiveOn & fw_.isHDRExtensionSupported()) { // if fullscreen exclusive is on (enabled & supported & turned on) only. HDR does not work in windowed mode properly unless "Windows HDR" is toggled on in settings for the users computer
			  for (auto& fmt : fmts) {									// But there is no way of query the state of "Windows HDR" being on for the application. So if Windows HDR is off, then the windowed mode with hdr on here would be incorrect.
																		// however, if fullscreen exclusive is on (default), then HDR can be properly controlled within the application.
				  if (fmt.surfaceFormat.format == vk::Format::eA2R10G10B10UnormPack32 && fmt.surfaceFormat.colorSpace == vk::ColorSpaceKHR::eHdr10St2084EXT) {
					  swapchainImageFormat_ = fmt.surfaceFormat.format;
					  swapchainColorSpace_ = fmt.surfaceFormat.colorSpace;
					  bHDROn = true;
					  break;
				  }
			  }
		  }

		  // if no 10 bit target exists,
		  if (!bHDROn) { // search for preferred 8bit target
			  for (auto& fmt : fmts) {
				  if (fmt.surfaceFormat.format == vk::Format::eB8G8R8A8Unorm && fmt.surfaceFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
					  swapchainImageFormat_ = fmt.surfaceFormat.format;
					  swapchainColorSpace_ = fmt.surfaceFormat.colorSpace;
					  break;
				  }
			  }
		  }
	  }

	  if (swapchainImageFormat_ == vk::Format::eA2R10G10B10UnormPack32 && swapchainColorSpace_ == vk::ColorSpaceKHR::eHdr10St2084EXT) {
		  fmt::print(fg(fmt::color::lime_green), "10bit Backbuffer - HDR10");
	  }
	  else if (swapchainImageFormat_ == vk::Format::eB8G8R8A8Unorm && swapchainColorSpace_ == vk::ColorSpaceKHR::eSrgbNonlinear) {
		  fmt::print(fg(fmt::color::lime_green), "8bit Backbuffer - SRGB");
	  }
	  else {
		  fmt::print(fg(fmt::color::red), "[FAIL] No compatible backbuffer format / color space found!");
		  return(false); // this is critical, would make everything extremely washed out or extremely dark, fail launch completely so game never pubicly looks like this
	  }

	  auto pms = physicalDevice_.getSurfacePresentModes2EXT(surfaceInfo).value;
	  vk::PresentModeKHR swapchainPresentMode = pms[0]; // default to first available

	  if (vsyncDisabled_)
	  {
		  if (std::find(pms.begin(), pms.end(), vk::PresentModeKHR::eImmediate) != pms.end()) {
			  swapchainPresentMode = vk::PresentModeKHR::eImmediate;
			  fmt::print(fg(fmt::color::lime_green), " - (VSYNC forced off) - ");
		  }
		  else {
			  fmt::print(fg(fmt::color::lime_green), " - (VSYNC could not be forced off) - ");
		  }
	  }
	  else {
		  // in order of preference
		  if (std::find(pms.begin(), pms.end(), vk::PresentModeKHR::eMailbox) != pms.end()) {
			  swapchainPresentMode = vk::PresentModeKHR::eMailbox;
		  }
		  else if (std::find(pms.begin(), pms.end(), vk::PresentModeKHR::eFifoRelaxed) != pms.end()) {
			  swapchainPresentMode = vk::PresentModeKHR::eFifoRelaxed;
		  }
		  else if (std::find(pms.begin(), pms.end(), vk::PresentModeKHR::eFifo) != pms.end()) {
			  swapchainPresentMode = vk::PresentModeKHR::eFifo;
		  }
	  }

	  vk::SwapchainCreateInfoKHR swapinfo{};
	  std::array<uint32_t, 1> const queueFamilyIndices = { graphicsQueueFamilyIndex_ };

	  uint32_t const imageCount = SFM::min(MAX_IMAGE_VIEWS, surfaceCaps.surfaceCapabilities.maxImageCount);
	  
	  fmt::print(fg(fmt::color::lime_green), " < {:s} >" "  Double-Buffered\n", vk::to_string(swapchainPresentMode));

	  swapinfo.surface = surface_;
	  swapinfo.minImageCount = imageCount;		
	  swapinfo.imageFormat = swapchainImageFormat_;
	  swapinfo.imageColorSpace = swapchainColorSpace_;
	  swapinfo.imageExtent = surfaceCaps.surfaceCapabilities.currentExtent;
	  swapinfo.imageArrayLayers = 1;
	  swapinfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
	  swapinfo.imageSharingMode = vk::SharingMode::eExclusive;	// best to use Exclusive Sharing Mode for performance optimizaion: https://gpuopen.com/vulkan-and-doom/ 
	  swapinfo.queueFamilyIndexCount = 1;
	  swapinfo.pQueueFamilyIndices = queueFamilyIndices.data();
	  swapinfo.preTransform = surfaceCaps.surfaceCapabilities.currentTransform;
	  swapinfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
	  swapinfo.presentMode = swapchainPresentMode;
	  swapinfo.clipped = VK_TRUE;
	  swapinfo.oldSwapchain = (swapchain_ ? *swapchain_ : vk::SwapchainKHR{});

#if defined(FULLSCREEN_EXCLUSIVE) && defined(VK_EXT_full_screen_exclusive)
	  if (bFullScreenExclusiveOn) {
		  swapinfo.pNext = &full_screen_exclusive;
	  }
#endif

	  // release old image views if they exists
	  for (auto& iv : imageViews_) {
		  device_.destroyImageView(iv);
	  }
	  images_.clear();
	  images_.reserve(MAX_IMAGE_VIEWS);
	  imageViews_.reserve(MAX_IMAGE_VIEWS);

	  vk::UniqueSwapchainKHR swapchain = device_.createSwapchainKHRUnique(swapinfo).value;
	  if (swapchain_) {
		  device_.destroySwapchainKHR(*swapchain_);  
	  }
	  swapchain_.swap(swapchain);

	  images_ = device_.getSwapchainImagesKHR(*swapchain_).value;
	  for (auto& img : images_) {
		  vk::ImageViewCreateInfo ci{};
		  ci.image = img;
		  ci.viewType = vk::ImageViewType::e2D;
		  ci.format = swapchainImageFormat_;
		  ci.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
		  imageViews_.emplace_back(device_.createImageView(ci).value);
	  }

	  return(true);
  }

  void initializeCheckerboardStencilBufferImages(vk::CommandPool const& __restrict commandPool, vk::Queue const& __restrict queue)
  {
	  struct sCHECKERBOARDDATA
	  {
		  vk::UniquePipelineLayout		pipelineLayout;
		  vk::Pipeline					pipeline;
		  vk::UniqueDescriptorSetLayout	descLayout;
		  std::vector<vk::DescriptorSet>sets;

		  sCHECKERBOARDDATA()
		  {}

		  ~sCHECKERBOARDDATA()
		  {
			  pipelineLayout.release();
			  sets.clear(); sets.shrink_to_fit();
			  descLayout.release();
		  }
	  } checkData[eCheckerboard::_size()];

	  // temporary renderpass ###############################################################################################################
	  vk::UniqueRenderPass renderPass_checkered;

	  // Build the renderpass using two attachments, colour and depth/stencil. (regular rendering pass)
	  {
		  vku::RenderpassMaker rpm;

		  // The stencil attachment.
		  rpm.attachmentBegin(stencilCheckerboard_[0].format());		// 0  // same format used for even and odd
		  rpm.attachmentSamples(vk::SampleCountFlagBits::e1);
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eStore);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eDepthStencilReadOnlyOptimal);

		  // A subpass to render using the above attachment
		  rpm.subpassBegin(vk::PipelineBindPoint::eGraphics);
		  rpm.subpassDepthStencilAttachment(vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal, 0);	// optimal format (read/write) during subpass

		  // A dependency to reset the layout of both attachments.
		  rpm.dependencyBegin(VK_SUBPASS_EXTERNAL, 0);
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eTopOfPipe);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests);
		  rpm.dependencySrcAccessMask((vk::AccessFlags)0);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eDepthStencilAttachmentWrite);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  rpm.dependencyBegin(0, VK_SUBPASS_EXTERNAL);
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eDepthStencilAttachmentWrite);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eDepthStencilAttachmentRead);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  // Use the maker object to construct the vulkan object
		  renderPass_checkered = rpm.createUnique(device_);
	  }

	  vku::ShaderModule const vert_{ device_, SHADER_PATH SHADER_POSTQUAD };

	  for (uint32_t odd = 0; odd < eCheckerboard::_size(); ++odd)
	  {
		  // temporary framebuffer ###############################################################################################################
		  vk::UniqueFramebuffer frameBuffer_checkered;

		  point2D const frameBufferSz(width_,height_);
		  point2D_t const downResFrameBufferSz(vku::getDownResolution(frameBufferSz));

		  vk::ImageView const attachments[1] = { stencilCheckerboard_[odd].imageView() };
		  vk::FramebufferCreateInfo const fbci{ {}, *renderPass_checkered, _countof(attachments), attachments, uint32_t(downResFrameBufferSz.x), uint32_t(downResFrameBufferSz.y), 1 };
		  frameBuffer_checkered = device_.createFramebufferUnique(fbci).value;


		  // temporary pipeline ###################################################################################################################
		  vku::ShaderModule const frag_{ device_, SHADER_CHECKERBOARD(SHADER_PATH, odd) };

		  // Build a template for descriptor sets that use these shaders.
		  vku::DescriptorSetLayoutMaker	dslm;
		  checkData[odd].descLayout = dslm.createUnique(device_);
		  // We need to create a descriptor set to tell the shader where
		  // our buffers are.
		  vku::DescriptorSetMaker			dsm;
		  dsm.layout(*checkData[odd].descLayout);
		  checkData[odd].sets = dsm.create(device_, fw_.descriptorPool());

		  // Make a default pipeline layout. This shows how pointers
		  // to resources are layed out.
		  // 
		  vku::PipelineLayoutMaker		plm;
		  plm.descriptorSetLayout(*checkData[odd].descLayout);
		  checkData[odd].pipelineLayout = plm.createUnique(device_);

		  // Make a pipeline to use the vertex format and shaders.
		  vku::PipelineMaker pm(uint32_t(downResFrameBufferSz.x), uint32_t(downResFrameBufferSz.y));
		  pm.shader(vk::ShaderStageFlagBits::eVertex, vert_);
		  pm.shader(vk::ShaderStageFlagBits::eFragment, frag_);

		  pm.depthCompareOp(vk::CompareOp::eAlways);
		  pm.depthClampEnable(VK_FALSE); // must be false
		  pm.depthTestEnable(VK_FALSE);
		  pm.depthWriteEnable(VK_FALSE);
		  // ################################
		  pm.stencilTestEnable(VK_TRUE); // only stencil
		  vk::StencilOpState const stencilOp{ 
			  /*vk::StencilOp failOp_ =*/ vk::StencilOp::eKeep,
			  /*vk::StencilOp passOp_ =*/ vk::StencilOp::eReplace,
			  /*vk::StencilOp depthFailOp_ =*/ vk::StencilOp::eKeep,
			  /*vk::CompareOp compareOp_ =*/ vk::CompareOp::eAlways,
			  /*uint32_t compareMask_ =*/ (uint32_t)0xff,
			  /*uint32_t writeMask_ =*/ (uint32_t)0xff,
			  /*uint32_t reference_ =*/ (uint32_t)STENCIL_CHECKER_REFERENCE
		  };
		  pm.front(stencilOp);
		  pm.back(stencilOp);

		  // ################################
		  pm.cullMode(vk::CullModeFlagBits::eFront);
		  pm.frontFace(vk::FrontFace::eCounterClockwise);

		  pm.blendBegin(VK_FALSE);
		  pm.blendColorWriteMask((vk::ColorComponentFlagBits)0); // no color writes

		  // Create a pipeline using a renderPass
		  pm.subPass(0);
		  pm.rasterizationSamples(vk::SampleCountFlagBits::e1);

		  auto& cache = fw_.pipelineCache();
		  checkData[odd].pipeline = pm.create(device_, cache, *checkData[odd].pipelineLayout, *renderPass_checkered);


		  // render checkerboard into stencil #########################################################################################################
		  // must wait, device is only locked inside lambda. So any operations occuring on the main thread involving the device or queue
		  // could be dangerous. This just so happens to occur at one of those times during init.
		  vku::executeImmediately<true>(device_, commandPool, queue, [&](vk::CommandBuffer cb) {
			
			  VKU_SET_CMD_BUFFER_LABEL(cb, vkNames::CommandBuffer::CHECKERBOARD);

			  point2D const frameBufferSz(width_, height_);
			  point2D_t const downResFrameBufferSz(vku::getDownResolution(frameBufferSz));

			  vk::RenderPassBeginInfo rpbi;
			  rpbi.renderPass = *renderPass_checkered;
			  rpbi.framebuffer = *frameBuffer_checkered;
			  rpbi.renderArea = vk::Rect2D{ {0, 0}, {uint32_t(downResFrameBufferSz.x), uint32_t(downResFrameBufferSz.y)} };
			  rpbi.clearValueCount = 0;
			  rpbi.pClearValues = nullptr;

			  cb.beginRenderPass(rpbi, vk::SubpassContents::eInline);

			  cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *checkData[odd].pipelineLayout, 0, checkData[odd].sets[0], nullptr);
			  cb.bindPipeline(vk::PipelineBindPoint::eGraphics, checkData[odd].pipeline);
			  cb.draw(3, 1, 0, 0);

			  cb.endRenderPass();
		  });

		  frameBuffer_checkered.release();

	  } // for

	  renderPass_checkered.release();

	  // final layout is transitioned for both odd and even stencil checkerboard at end of renderpass to eDepthStencilReadOnlyOptimal by renderpass automatically
	  // no further transitions needed or allowed
  }

  public:

  void init(const vk::Instance &instance, const vk::Device &device, const vk::PhysicalDevice &physicalDevice, uint32_t const graphicsQueueFamilyIndex, uint32_t const computeQueueFamilyIndex, uint32_t const transferQueueFamilyIndex, vk::SurfaceKHR const surface, HMONITOR const& monitor, bool const bVsyncDisabled) {
	  
	  surface_ = surface;
	  instance_ = instance;
	  device_ = device;
	  
	  // neccessary to cache for hot swapchain recreation
	  physicalDevice_ = physicalDevice;
	  graphicsQueueFamilyIndex_ = graphicsQueueFamilyIndex;
	  monitor_ = monitor;
	  vsyncDisabled_ = bVsyncDisabled;

	  { // major bugfix: graphics queue is now ALWAYS the presenting queue. It has the lowest latency for present, allows overlap of dma transfers queues used
		  // in beginning of vku render as a transfer queue is no longer tied up with "presenting". It also has the greatest compatibility among graphics cards.
		  // ** do not change **

		  if (!physicalDevice_.getSurfaceSupportKHR(graphicsQueueFamilyIndex_, surface_).value) {  /// required by validation to atleast check queue against surface support
			  FMT_LOG_FAIL(GPU_LOG, "Graphics Queue does not support presentable surface requirements! ( {:d} )", graphicsQueueFamilyIndex_);
		  }
	  }
	  // save queues
	  graphicsQueue_    = device.getQueue(graphicsQueueFamilyIndex, 0);
	  computeQueue_[0]  = device.getQueue(computeQueueFamilyIndex, 0);
	  computeQueue_[1]  = device.getQueue(computeQueueFamilyIndex, 1);
	  transferQueue_[0] = device.getQueue(transferQueueFamilyIndex, 0);
	  transferQueue_[1] = device.getQueue(transferQueueFamilyIndex, 1);

	  // initial creation of swapchain
	  if (!recreateSwapChain()) {
		  FMT_LOG_FAIL(GPU_LOG, "Major swapchain fail");
		  return;
	  }

	  {
		  vk::CommandPoolCreateInfo cpci{ vk::CommandPoolCreateFlagBits::eTransient, graphicsQueueFamilyIndex };
		  commandPool_[eCommandPools::TRANSIENT_POOL] = device.createCommandPoolUnique(cpci).value;
	  }

	  point2D const frameBufferSz(width_, height_);
	  point2D_t const downResFrameBufferSz(vku::getDownResolution(frameBufferSz));

	  {
		  // only for simplifying this critical section / initialization of all color attachments, depth attachments for readability
		  auto const& __restrict transientCommandPool = *commandPool_[eCommandPools::TRANSIENT_POOL];

		  colorImage_ = vku::ColorAttachmentImage(device, width_, height_, vku::DefaultSampleCount, transientCommandPool, graphicsQueue_, false, false, false, vk::Format::eB8G8R8A8Unorm);	// not sampled, not inputattachment, not copyable
		  lastColorImage_ = vku::ColorAttachmentImage(device, width_, height_, vk::SampleCountFlagBits::e1, transientCommandPool, graphicsQueue_, true, false, false, vk::Format::eB8G8R8A8Unorm);	// is sampled, not inputattachment, not copyable
		  depthImage_ = vku::DepthAttachmentImage(device, width_, height_, vku::DefaultSampleCount, transientCommandPool, graphicsQueue_, false, true);  // is inputattachment
		  VKU_SET_OBJECT_NAME(vk::ObjectType::eImage, (VkImage)colorImage_.image(), vkNames::Image::colorImage);
		  VKU_SET_OBJECT_NAME(vk::ObjectType::eImage, (VkImage)lastColorImage_.image(), vkNames::Image::lastColorImage);
		  VKU_SET_OBJECT_NAME(vk::ObjectType::eImage, (VkImage)depthImage_.image(), vkNames::Image::depthImage);

		  mouseImage_.multisampled = vku::ColorAttachmentImage(device, width_, height_, vku::DefaultSampleCount, transientCommandPool, graphicsQueue_, false, false, false, vk::Format::eR16G16Unorm);	// not sampled, not inputattachment, not copyable
		  mouseImage_.resolved = vku::ColorAttachmentImage(device, width_, height_, vk::SampleCountFlagBits::e1, transientCommandPool, graphicsQueue_, false, false, true, vk::Format::eR16G16Unorm);	// not sampled, not inputattachment, copyable
		  VKU_SET_OBJECT_NAME(vk::ObjectType::eImage, (VkImage)mouseImage_.multisampled.image(), vkNames::Image::mouseImage_multisampled);
		  VKU_SET_OBJECT_NAME(vk::ObjectType::eImage, (VkImage)mouseImage_.resolved.image(), vkNames::Image::mouseImage_resolved);

		  depthImageResolve_[0] = vku::DepthImage(device, width_, height_, transientCommandPool, graphicsQueue_, true, false);  // depth only image - is colorattachment
		  depthImageResolve_[1] = vku::DepthImage(device, uint32_t(downResFrameBufferSz.x), uint32_t(downResFrameBufferSz.y), transientCommandPool, graphicsQueue_, false, true);  // depth only image - is storage
		  stencilCheckerboard_[0] = vku::StencilAttachmentImage(device, uint32_t(downResFrameBufferSz.x), uint32_t(downResFrameBufferSz.y), transientCommandPool, graphicsQueue_);
		  stencilCheckerboard_[1] = vku::StencilAttachmentImage(device, uint32_t(downResFrameBufferSz.x), uint32_t(downResFrameBufferSz.y), transientCommandPool, graphicsQueue_);

		  for (uint32_t i = 0; i < 2; ++i) {
			  VKU_SET_OBJECT_NAME(vk::ObjectType::eImage, (VkImage)depthImageResolve_[i].image(), vkNames::Image::depthImageResolve);
			  VKU_SET_OBJECT_NAME(vk::ObjectType::eImage, (VkImage)stencilCheckerboard_[i].image(), vkNames::Image::stencilCheckerboard);
		  }

		  // vk::ImageUsageFlagBits::eTransferDst no longer needed as temporal blending has been enabled for reconstruction (no clears!)
		  colorVolumetricImage_.checkered = vku::TextureImageStorage2D(vk::ImageUsageFlagBits::eSampled /*| vk::ImageUsageFlagBits::eTransferDst*/, device, uint32_t(downResFrameBufferSz.x), uint32_t(downResFrameBufferSz.y), 1U, vk::SampleCountFlagBits::e1, vk::Format::eB8G8R8A8Unorm, false, true);  // not host image, is dedicated
		  colorVolumetricImage_.resolved = vku::ColorAttachmentImage(device, uint32_t(downResFrameBufferSz.x), uint32_t(downResFrameBufferSz.y), vk::SampleCountFlagBits::e1, transientCommandPool, graphicsQueue_, true, false, false, vk::Format::eB8G8R8A8Unorm);  // is sampled, not inputattachment, not copyable
		  colorVolumetricImage_.upsampled = vku::ColorAttachmentImage(device, width_, height_, vk::SampleCountFlagBits::e1, transientCommandPool, graphicsQueue_, false, true, false, vk::Format::eB8G8R8A8Unorm);  // not sampled, is inputattachment, not copyable
		  
		  VKU_SET_OBJECT_NAME(vk::ObjectType::eImage, (VkImage)colorVolumetricImage_.checkered.image(), vkNames::Image::colorVolumetricImage_checkered);
		  VKU_SET_OBJECT_NAME(vk::ObjectType::eImage, (VkImage)colorVolumetricImage_.resolved.image(), vkNames::Image::colorVolumetricImage_resolved);
		  VKU_SET_OBJECT_NAME(vk::ObjectType::eImage, (VkImage)colorVolumetricImage_.upsampled.image(), vkNames::Image::colorVolumetricImage_upsampled);

		  // reflections are captured in screen space (2D) on a half-res render target
		  colorReflectionImage_.checkered = vku::TextureImageStorage2D(vk::ImageUsageFlagBits::eSampled /*| vk::ImageUsageFlagBits::eTransferDst*/, device, uint32_t(downResFrameBufferSz.x), uint32_t(downResFrameBufferSz.y), 1U, vk::SampleCountFlagBits::e1, vk::Format::eB8G8R8A8Unorm, false, true);  // not host image, is dedicated
		  colorReflectionImage_.resolved = vku::ColorAttachmentImage(device, uint32_t(downResFrameBufferSz.x), uint32_t(downResFrameBufferSz.y), vk::SampleCountFlagBits::e1, transientCommandPool, graphicsQueue_, true, false, false, vk::Format::eB8G8R8A8Unorm);  // is sampled, not inputattachment, not copyable
		  colorReflectionImage_.upsampled = vku::ColorAttachmentImage(device, width_, height_, vk::SampleCountFlagBits::e1, transientCommandPool, graphicsQueue_, true, true, false, vk::Format::eB8G8R8A8Unorm);  // is sampled, is inputattachment, not copyable

		  VKU_SET_OBJECT_NAME(vk::ObjectType::eImage, (VkImage)colorReflectionImage_.checkered.image(), vkNames::Image::colorReflectionImage_checkered);
		  VKU_SET_OBJECT_NAME(vk::ObjectType::eImage, (VkImage)colorReflectionImage_.resolved.image(), vkNames::Image::colorReflectionImage_resolved);
		  VKU_SET_OBJECT_NAME(vk::ObjectType::eImage, (VkImage)colorReflectionImage_.upsampled.image(), vkNames::Image::colorReflectionImage_upsampled);

		  for (uint32_t i = 0 ; i < 2 ; ++i) {
			  guiImage_.multisampled[i] = vku::ColorAttachmentImage(device, width_, height_, vku::DefaultSampleCount, transientCommandPool, graphicsQueue_, false, false, false);	// not sampled, not inputattachment, not copyable
			  guiImage_.resolved[i] = vku::ColorAttachmentImage(device, width_, height_, vk::SampleCountFlagBits::e1, transientCommandPool, graphicsQueue_, false, true, false);	// not sampled, is inputattachment, not copyable

			  VKU_SET_OBJECT_NAME(vk::ObjectType::eImage, (VkImage)guiImage_.multisampled[i].image(), vkNames::Image::guiImage);
			  VKU_SET_OBJECT_NAME(vk::ObjectType::eImage, (VkImage)guiImage_.resolved[i].image(), vkNames::Image::guiImage);
		  }

		  offscreenImage_.multisampled = vku::ColorAttachmentImage(device, width_, height_, vku::DefaultSampleCount, transientCommandPool, graphicsQueue_, false, false, false);	// not sampled, not inputattachment, not copyable
		  offscreenImage_.resolved = vku::ColorAttachmentImage(device, width_, height_, vk::SampleCountFlagBits::e1, transientCommandPool, graphicsQueue_, true, false, true);	// sampled, not inputattachment, copyable

		  VKU_SET_OBJECT_NAME(vk::ObjectType::eImage, (VkImage)offscreenImage_.multisampled.image(), vkNames::Image::offscreenImage);
		  VKU_SET_OBJECT_NAME(vk::ObjectType::eImage, (VkImage)offscreenImage_.resolved.image(), vkNames::Image::offscreenImage);

		  
		  vku::executeImmediately<true>(device, transientCommandPool, graphicsQueue_, [&](vk::CommandBuffer cb) {

			  // never changes layout setup : //
			  colorVolumetricDownResCheckeredImage().setLayout(cb, vk::ImageLayout::eGeneral);
			  colorReflectionDownResCheckeredImage().setLayout(cb, vk::ImageLayout::eGeneral);

			  // reflection (upsampled) start up requirement
			  colorReflectionImage().setLayout(cb, vk::ImageLayout::eShaderReadOnlyOptimal);

			  // gui images start up requirement
			  guiImage(0).setLayout(cb, vk::ImageLayout::eShaderReadOnlyOptimal);
			  guiImage(1).setLayout(cb, vk::ImageLayout::eShaderReadOnlyOptimal);

			  });
	  }
	  // Build the renderpass using two attachments, colour and depth/stencil. (regular rendering pass)
	  {
		  vku::RenderpassMaker rpm;

		  // **** SUBPASS 0 - Regular rendering - ZONLY No Color Writes & Clear Masks Alpha Writes (1st color attachment) //
		 
		  // The depth/stencil attachment.
		  rpm.attachmentBegin(depthImage_.format());		// 0
		  rpm.attachmentSamples(vku::DefaultSampleCount);
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eClear);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eStore);  // used in later renderpass
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eUndefined);					// undefined should be used to reset beginning state if load op is clear
		  rpm.attachmentFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal); 

		  // The first colour attachment. (only alpha writes enabled in zpass for clearmasks)
		  rpm.attachmentBegin(colorImage_.format());	   // 1
		  rpm.attachmentSamples(vku::DefaultSampleCount);
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eClear);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eStore);
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eUndefined);					// undefined should be used to reset beginning state if load op is clear
		  rpm.attachmentFinalLayout(vk::ImageLayout::eColorAttachmentOptimal);

		  // The second colour attachment.				     		// 2
		  rpm.attachmentBegin(mouseImage_.multisampled.format()); 
		  rpm.attachmentSamples(vku::DefaultSampleCount);								  
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eClear);							  
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eDontCare);		// not required to store - multisampled image is fully transient for this *renderpass*
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare); // is used only in subpass 2 for resolving the multisampled image
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eUndefined);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eColorAttachmentOptimal);

		  // A subpass to render using the above attachment
		  rpm.subpassBegin(vk::PipelineBindPoint::eGraphics);
		  rpm.subpassDepthStencilAttachment(vk::ImageLayout::eDepthStencilAttachmentOptimal, 0);	// optimal format (read/write) during subpass
		  rpm.subpassColorAttachment(vk::ImageLayout::eColorAttachmentOptimal, 1);
		  rpm.subpassColorAttachment(vk::ImageLayout::eColorAttachmentOptimal, 2);

		  // **** SUBPASS 1 - Depth buffer custom resolve //

		  // The depth/stencil attachment.
		  rpm.attachmentBegin(depthImage_.format());		// 3
		  rpm.attachmentSamples(vku::DefaultSampleCount);
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eLoad);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eStore); // used in later renderpass
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eDepthStencilReadOnlyOptimal);	// depth shall remain readonly for the rest of the frame

		  // The only colour attachment.
		  rpm.attachmentBegin(depthImageResolve_[0].format());		// 4
		  rpm.attachmentSamples(vk::SampleCountFlagBits::e1);
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eStore);
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eUndefined);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

		  // A subpass to render using the above two attachments.
		  rpm.subpassBegin(vk::PipelineBindPoint::eGraphics);
		  rpm.subpassInputAttachment(vk::ImageLayout::eDepthStencilReadOnlyOptimal, 3);
		  rpm.subpassColorAttachment(vk::ImageLayout::eColorAttachmentOptimal, 4);

		  // **** SUBPASS 2 - Mouse Image resolve //

		  // The second colour attachment.				     		// 5
		  rpm.attachmentBegin(mouseImage_.multisampled.format());
		  rpm.attachmentSamples(vku::DefaultSampleCount);
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eLoad);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eColorAttachmentOptimal);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eColorAttachmentOptimal);

		  // The resolved second colour attachment.				     // 6
		  rpm.attachmentBegin(mouseImage_.resolved.format());
		  rpm.attachmentSamples(vk::SampleCountFlagBits::e1);
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eStore);				// store required for resolved image
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eUndefined);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eTransferSrcOptimal);

		  // A subpass to render using the above two attachments.
		  rpm.subpassBegin(vk::PipelineBindPoint::eGraphics);
		  rpm.subpassColorAttachment(vk::ImageLayout::eColorAttachmentOptimal, 5);
		  rpm.subpassResolveAttachment(vk::ImageLayout::eColorAttachmentOptimal, 6);



		  // A dependency to reset the layout of both attachments.
		  rpm.dependencyBegin(VK_SUBPASS_EXTERNAL, 0);
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eTopOfPipe);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eEarlyFragmentTests| vk::PipelineStageFlagBits::eLateFragmentTests);
		  rpm.dependencySrcAccessMask((vk::AccessFlags)0);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eDepthStencilAttachmentRead|vk::AccessFlagBits::eDepthStencilAttachmentWrite);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  rpm.dependencyBegin(VK_SUBPASS_EXTERNAL, 0);
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eTopOfPipe);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencySrcAccessMask((vk::AccessFlags)0);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  rpm.dependencyBegin(VK_SUBPASS_EXTERNAL, 1);
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eTopOfPipe);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencySrcAccessMask((vk::AccessFlags)0);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  rpm.dependencyBegin(0, 1);
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eFragmentShader);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eDepthStencilAttachmentRead | vk::AccessFlagBits::eDepthStencilAttachmentWrite);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eInputAttachmentRead);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  rpm.dependencyBegin(1, VK_SUBPASS_EXTERNAL);
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eFragmentShader);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eInputAttachmentRead);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  rpm.dependencyBegin(1, VK_SUBPASS_EXTERNAL);
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eFragmentShader);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eInputAttachmentRead);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eDepthStencilAttachmentRead);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  // mouse resolve
		  rpm.dependencyBegin(0, 2);
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eColorAttachmentRead);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  rpm.dependencyBegin(2, VK_SUBPASS_EXTERNAL);
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eTransfer);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eColorAttachmentRead);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eTransferRead);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  // Use the maker object to construct the vulkan object
		  zPass_ = rpm.createUnique(device);

		  VKU_SET_OBJECT_NAME(vk::ObjectType::eRenderPass, (VkRenderPass)*zPass_, vkNames::Renderpass::ZPASS);
	  }

	  framebuffers_[eFrameBuffers::DEPTH] = new vk::UniqueFramebuffer[imageViews_.size()];
	  for (int i = 0; i != imageViews_.size(); ++i) {
		  vk::ImageView const attachments[7] = { depthImage_.imageView()/*cleared*/, colorImage_.imageView()/*cleared*/, mouseImage_.multisampled.imageView()/*cleared*/, 
												 depthImage_.imageView(), depthImageResolve_[0].imageView(),
												 mouseImage_.multisampled.imageView(), mouseImage_.resolved.imageView()
											   };
		  vk::FramebufferCreateInfo const fbci{ {}, *zPass_, _countof(attachments), attachments, width_, height_, 1 };
		  framebuffers_[eFrameBuffers::DEPTH][i] = std::move(device.createFramebufferUnique(fbci).value);

		  VKU_SET_OBJECT_NAME(vk::ObjectType::eFramebuffer, (VkFramebuffer)*framebuffers_[eFrameBuffers::DEPTH][i], vkNames::FrameBuffer::DEPTH);
	  }

	  //  (down - rezzed - resolution pass) **vku::DOWN_RES_FACTOR sets resolution factor of framebuffer
	  {
		  vku::RenderpassMaker rpm;

		  // SUBPASS 0 - Raymarch

		  // The stencil (checkerboard) attachment.
		  rpm.attachmentBegin(stencilCheckerboard_[0].format());				// 0
		  rpm.attachmentSamples(vk::SampleCountFlagBits::e1);
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eLoad);				// stencil only
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eDepthStencilReadOnlyOptimal);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eDepthStencilReadOnlyOptimal);

		  // The input attachment.
		  rpm.attachmentBegin(depthImageResolve_[1].format());					// 1
		  rpm.attachmentSamples(vk::SampleCountFlagBits::e1);
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eLoad);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eDontCare); // not used outside renderpass
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eGeneral);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

		  // A subpass to render using the above attachment.
		  rpm.subpassBegin(vk::PipelineBindPoint::eGraphics);
		  rpm.subpassDepthStencilAttachment(vk::ImageLayout::eDepthStencilReadOnlyOptimal, 0);
		  rpm.subpassInputAttachment(vk::ImageLayout::eShaderReadOnlyOptimal, 1);
		  
		  // *** SUBPASS 1 - Resolve Raymarch outputs (volumetric and reflection)

		  // The colour attachment.
		  rpm.attachmentBegin(colorVolumetricImage_.resolved.format());					// 2
		  rpm.attachmentSamples(vk::SampleCountFlagBits::e1);
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eStore);
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eUndefined);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eShaderReadOnlyOptimal);


		  // The colour attachment.
		  rpm.attachmentBegin(colorReflectionImage_.resolved.format());					// 3
		  rpm.attachmentSamples(vk::SampleCountFlagBits::e1);
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eStore);
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eUndefined);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eShaderReadOnlyOptimal);


		  // A subpass to render using the above attachment.
		  rpm.subpassBegin(vk::PipelineBindPoint::eGraphics);
		  rpm.subpassColorAttachment(vk::ImageLayout::eColorAttachmentOptimal, 2);
		  rpm.subpassColorAttachment(vk::ImageLayout::eColorAttachmentOptimal, 3);

		  // stencil test readonly dependency
		  rpm.dependencyBegin(VK_SUBPASS_EXTERNAL, 0);
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eDepthStencilAttachmentRead);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eDepthStencilAttachmentRead);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  // depth resolved, half resolution dependency
		  rpm.dependencyBegin(VK_SUBPASS_EXTERNAL, 0);
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eFragmentShader);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eFragmentShader);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eShaderWrite);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eInputAttachmentRead);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  // transition image stored in general to input attachments
		  rpm.dependencyBegin(0, 1);
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eFragmentShader);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eFragmentShader);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eShaderWrite);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eShaderRead);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  // transition color attachments to shaderreadonly
		  rpm.dependencyBegin(1, VK_SUBPASS_EXTERNAL);
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eFragmentShader);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eShaderRead);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  // Use the maker object to construct the vulkan object
		  downPass_ = rpm.createUnique(device);

		  VKU_SET_OBJECT_NAME(vk::ObjectType::eRenderPass, (VkRenderPass)*downPass_, vkNames::Renderpass::DOWNPASS);
	  }

	  // even, odd     -or-    even, odd, even, odd            - imageViews must be even for this to wrap around correctly
	  framebuffers_[eFrameBuffers::HALF_COLOR_ONLY] = new vk::UniqueFramebuffer[imageViews_.size()];
	  for (int i = 0; i != imageViews_.size(); ++i) {

		  int const odd = i & 1; // auto-magical alternating checkerboard stencil usage

												 // subpass 0
		  vk::ImageView const attachments[4] = { stencilCheckerboard_[odd].imageView(), depthImageResolve_[1].imageView(),
												 // subpass 1
												 colorVolumetricImage_.resolved.imageView(), colorReflectionImage_.resolved.imageView()
											   };
		  vk::FramebufferCreateInfo const fbci{ {}, *downPass_, _countof(attachments), attachments, uint32_t(downResFrameBufferSz.x), uint32_t(downResFrameBufferSz.y), 1 };
		  framebuffers_[eFrameBuffers::HALF_COLOR_ONLY][i] = std::move(device.createFramebufferUnique(fbci).value);

		  VKU_SET_OBJECT_NAME(vk::ObjectType::eFramebuffer, (VkFramebuffer)*framebuffers_[eFrameBuffers::HALF_COLOR_ONLY][i], vkNames::FrameBuffer::HALF_COLOR_ONLY);
	  }

	  
	  //  (up - rezzed - resolution pass)
	  {
		  vku::RenderpassMaker rpm;

		  // *** SUBPASS 0 - Upsampling of haLF res volumetric pass & reflection
		 // The colour attachment.
		  rpm.attachmentBegin(colorVolumetricImage_.upsampled.format());					// 0
		  rpm.attachmentSamples(vk::SampleCountFlagBits::e1);
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eStore);
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eUndefined);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

		  // The colour attachment.
		  rpm.attachmentBegin(colorReflectionImage_.upsampled.format());					// 1
		  rpm.attachmentSamples(vk::SampleCountFlagBits::e1);
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eStore);
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eUndefined);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

		  // The input attachment.
		  rpm.attachmentBegin(depthImageResolve_[0].format());					 // 2
		  rpm.attachmentSamples(vk::SampleCountFlagBits::e1);
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eLoad);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eDontCare); // not used outside renderpass
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

		  // A subpass to render using the above attachment.
		  rpm.subpassBegin(vk::PipelineBindPoint::eGraphics);
		  rpm.subpassColorAttachment(vk::ImageLayout::eColorAttachmentOptimal, 0);
		  rpm.subpassColorAttachment(vk::ImageLayout::eColorAttachmentOptimal, 1);
		  rpm.subpassInputAttachment(vk::ImageLayout::eShaderReadOnlyOptimal, 2);


		  rpm.dependencyBegin(VK_SUBPASS_EXTERNAL, 0); // In  upsampling dependent on resolved colors
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eFragmentShader);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eInputAttachmentRead);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  // depth resolved, full + half resolution dependency
		  rpm.dependencyBegin(VK_SUBPASS_EXTERNAL, 0);
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eFragmentShader);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eFragmentShader);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eShaderWrite);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eInputAttachmentRead);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  // transition color attachments to input attachments for next renderpass
		  rpm.dependencyBegin(0, VK_SUBPASS_EXTERNAL);
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eFragmentShader);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eInputAttachmentRead);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  // Use the maker object to construct the vulkan object
		  upPass_ = rpm.createUnique(device);

		  VKU_SET_OBJECT_NAME(vk::ObjectType::eRenderPass, (VkRenderPass)*upPass_, vkNames::Renderpass::UPPASS);
	  }

	  framebuffers_[eFrameBuffers::FULL_COLOR_ONLY] = new vk::UniqueFramebuffer[imageViews_.size()];
	  for (int i = 0; i != imageViews_.size(); ++i) {
												// subpass 0
		  vk::ImageView const attachments[3] = { colorVolumetricImage_.upsampled.imageView(), colorReflectionImage_.upsampled.imageView(), depthImageResolve_[0].imageView()
											   };
		  vk::FramebufferCreateInfo const fbci{ {}, *upPass_, _countof(attachments), attachments, width_, height_, 1 };
		  framebuffers_[eFrameBuffers::FULL_COLOR_ONLY][i] = std::move(device.createFramebufferUnique(fbci).value);

		  VKU_SET_OBJECT_NAME(vk::ObjectType::eFramebuffer, (VkFramebuffer)*framebuffers_[eFrameBuffers::FULL_COLOR_ONLY][i], vkNames::FrameBuffer::FULL_COLOR_ONLY);
	  }



	  // Build the renderpass using one attachment, colour   (mid/intermediatte pass)
	  {
		  vku::RenderpassMaker rpm;

		  // *** SUBPASS 0 - Regular rendering //

		  // The only colour attachment.
		  rpm.attachmentBegin(colorImage_.format());							// 0
		  rpm.attachmentSamples(vku::DefaultSampleCount);
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eLoad);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eStore);
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eColorAttachmentOptimal);					
		  rpm.attachmentFinalLayout(vk::ImageLayout::eColorAttachmentOptimal);

		  // The depth/stencil attachment.
		  rpm.attachmentBegin(depthImage_.format());								// 1
		  rpm.attachmentSamples(vku::DefaultSampleCount);
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eLoad);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eDontCare); // used readonly
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eDepthStencilReadOnlyOptimal);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eDepthStencilReadOnlyOptimal);

		  // The input attachment. (reflection)
		  rpm.attachmentBegin(colorReflectionImage_.upsampled.format());							// 2
		  rpm.attachmentSamples(vk::SampleCountFlagBits::e1);
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eLoad);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

		  // A subpass to render using the above two attachments.
		  rpm.subpassBegin(vk::PipelineBindPoint::eGraphics);
		  rpm.subpassColorAttachment(vk::ImageLayout::eColorAttachmentOptimal, 0);
		  rpm.subpassDepthStencilAttachment(vk::ImageLayout::eDepthStencilReadOnlyOptimal, 1);	// optimal format (read/write) during subpass
		  rpm.subpassInputAttachment(vk::ImageLayout::eShaderReadOnlyOptimal, 2);



		  // *** SUBPASS 1 - Upsampled volumetric blend and resolve to lastColor //		

		  // The input attachment.
		  rpm.attachmentBegin(colorVolumetricImage_.upsampled.format());						// 3
		  rpm.attachmentSamples(vk::SampleCountFlagBits::e1);
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eLoad);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eDontCare);  // not used outside renderpass
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

		  // The resolved colour attachment.			// resolve for voxel transparency		// 4
		  rpm.attachmentBegin(lastColorImage_.format()); // output color attachment
		  rpm.attachmentSamples(vk::SampleCountFlagBits::e1);
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eStore);
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eUndefined);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

		  // A subpass to render using the above two attachments.
		  rpm.subpassBegin(vk::PipelineBindPoint::eGraphics);
		  rpm.subpassColorAttachment(vk::ImageLayout::eColorAttachmentOptimal, 0);
		  rpm.subpassInputAttachment(vk::ImageLayout::eShaderReadOnlyOptimal, 3);
		  rpm.subpassResolveAttachment(vk::ImageLayout::eColorAttachmentOptimal, 4);


		  rpm.dependencyBegin(VK_SUBPASS_EXTERNAL, 0); // In
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  rpm.dependencyBegin(VK_SUBPASS_EXTERNAL, 0);  // regular rendering dependent on reflections and real depth
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eFragmentShader);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eInputAttachmentRead);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  rpm.dependencyBegin(VK_SUBPASS_EXTERNAL, 0);
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eDepthStencilAttachmentRead);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eDepthStencilAttachmentRead);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  rpm.dependencyBegin(0, 1);  // upsample blend dependent color output finished ,,,
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  rpm.dependencyBegin(1, VK_SUBPASS_EXTERNAL); // Out  // resolved "lastColor" for transparency in overlay pass
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eFragmentShader);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eShaderRead);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  // Use the maker object to construct the vulkan object
		  midPass_ = rpm.createUnique(device);

		  VKU_SET_OBJECT_NAME(vk::ObjectType::eRenderPass, (VkRenderPass)*midPass_, vkNames::Renderpass::MIDPASS);
	  }

	  framebuffers_[eFrameBuffers::MID_COLOR_DEPTH] = new vk::UniqueFramebuffer[imageViews_.size()];
	  for (int i = 0; i != imageViews_.size(); ++i) {
												  
		  vk::ImageView const attachments[5] = { 
												// subpass 0
												 colorImage_.imageView(), depthImage_.imageView(), colorReflectionImage_.upsampled.imageView(),
												// subpass 1
												 colorVolumetricImage_.upsampled.imageView(), lastColorImage_.imageView()
											   };
		  vk::FramebufferCreateInfo const fbci{ {}, *midPass_, _countof(attachments), attachments, width_, height_, 1 };
		  framebuffers_[eFrameBuffers::MID_COLOR_DEPTH][i] = std::move(device.createFramebufferUnique(fbci).value);

		  VKU_SET_OBJECT_NAME(vk::ObjectType::eFramebuffer, (VkFramebuffer)*framebuffers_[eFrameBuffers::MID_COLOR_DEPTH][i], vkNames::FrameBuffer::MID_COLOR_DEPTH);
	  }



	  // Build the renderpass (overlay / transparency pass)
	  {
		  vku::RenderpassMaker rpm;

		  // *** 1st SUBPASS - Transparent Voxels
		  // The colour attachment.											// 0
		  rpm.attachmentBegin(colorImage_.format());
		  rpm.attachmentSamples(vku::DefaultSampleCount);
		  // Don't clear the framebuffer for overlay on top of main renderpass
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eLoad);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eDontCare);		// does not need to be stored, is fully transient in this renderpass, and not used later for reading in any subsequent renderpasses
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eColorAttachmentOptimal);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eColorAttachmentOptimal);

		  // The input attachment. (reflection)
		  rpm.attachmentBegin(colorReflectionImage_.upsampled.format());							// 1
		  rpm.attachmentSamples(vk::SampleCountFlagBits::e1);
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eLoad);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

		  // The depth/stencil attachment.
		  rpm.attachmentBegin(depthImage_.format());
		  rpm.attachmentSamples(vku::DefaultSampleCount);					// 2
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eLoad);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eDepthStencilReadOnlyOptimal);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eDepthStencilReadOnlyOptimal);

		  // A subpass to render using the above attachment.
		  rpm.subpassBegin(vk::PipelineBindPoint::eGraphics);
		  rpm.subpassColorAttachment(vk::ImageLayout::eColorAttachmentOptimal, 0);
		  rpm.subpassInputAttachment(vk::ImageLayout::eShaderReadOnlyOptimal, 1);
		  rpm.subpassDepthStencilAttachment(vk::ImageLayout::eDepthStencilReadOnlyOptimal, 2);

		  // *** 2nd SUBPASS - Isolated Resolve - Validation error if the lastColor is bound as sampler & attachment, even if its a resolve only

		 // The colour attachment.											// 3
		  rpm.attachmentBegin(colorImage_.format());
		  rpm.attachmentSamples(vku::DefaultSampleCount);
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eLoad);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eDontCare);		// does not need to be stored, is fully transient in this renderpass, and not used later for reading in any subsequent renderpasses
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eColorAttachmentOptimal);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eColorAttachmentOptimal);

		  // The resolved colour attachment.			// resolve for everything else that needs final color buffer w/o Post Postprocessing & GUI
		  rpm.attachmentBegin(lastColorImage_.format()); // output color attachment
		  rpm.attachmentSamples(vk::SampleCountFlagBits::e1);				// 4
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eStore);				// is sampled by next renderpass in post processing
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

		  // A subpass to render using the above attachment.
		  rpm.subpassBegin(vk::PipelineBindPoint::eGraphics);
		  rpm.subpassColorAttachment(vk::ImageLayout::eColorAttachmentOptimal, 3);
		  rpm.subpassResolveAttachment(vk::ImageLayout::eColorAttachmentOptimal, 4);

		  // *** 3rd SUBPASS - Nuklear GUI
		  // The colour attachment.
		  rpm.attachmentBegin(guiImage_.multisampled[0].format());
		  rpm.attachmentSamples(vku::DefaultSampleCount);
		  // must clear
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eClear);			// 5
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eUndefined);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eColorAttachmentOptimal);

		  // The resolved colour attachment.			// resolve for GUI
		  rpm.attachmentBegin(guiImage_.resolved[0].format());
		  rpm.attachmentSamples(vk::SampleCountFlagBits::e1);				// 6
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eStore);
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eUndefined);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

		  // A subpass to render using the above attachment.
		  rpm.subpassBegin(vk::PipelineBindPoint::eGraphics);
		  rpm.subpassColorAttachment(vk::ImageLayout::eColorAttachmentOptimal, 5);
		  rpm.subpassResolveAttachment(vk::ImageLayout::eColorAttachmentOptimal, 6);

		  // 2 dependency to reset the layout of attachment.
		  rpm.dependencyBegin(VK_SUBPASS_EXTERNAL, 0); // In
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  rpm.dependencyBegin(VK_SUBPASS_EXTERNAL, 0);
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eDepthStencilAttachmentRead);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eDepthStencilAttachmentRead);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  // reflection already in correct state (previous midpass), no dependency needed already input attachment - shaderreadonly

		  rpm.dependencyBegin(0, 1); // subpass -> subpass
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);
		  
		  rpm.dependencyBegin(1, VK_SUBPASS_EXTERNAL); // Out
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eFragmentShader);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eShaderRead);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  rpm.dependencyBegin(2, VK_SUBPASS_EXTERNAL); // Out
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eFragmentShader);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eShaderRead);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  // Use the maker object to construct the vulkan object
		  overlayPass_ = rpm.createUnique(device);

		  VKU_SET_OBJECT_NAME(vk::ObjectType::eRenderPass, (VkRenderPass)*overlayPass_, vkNames::Renderpass::OVERLAY);
	  }

	  framebuffers_[eFrameBuffers::COLOR_DEPTH] = new vk::UniqueFramebuffer[imageViews_.size()];
	  for (int i = 0; i != imageViews_.size(); ++i) {

		  int const odd = i & 1; // auto-magical alternating gui image target
											// 1st subpass																                                      
		  vk::ImageView attachments[7] = { colorImage_.imageView(), colorReflectionImage_.upsampled.imageView(), depthImage_.imageView(), 
											// 2nd subpass 		
											colorImage_.imageView(), lastColorImage_.imageView(),
											// 3rd subpass
										    guiImage_.multisampled[odd].imageView()/*cleared*/, guiImage_.resolved[odd].imageView() };

		  vk::FramebufferCreateInfo fbci{ {}, *overlayPass_, _countof(attachments), attachments, width_, height_, 1 };
		  framebuffers_[eFrameBuffers::COLOR_DEPTH][i] = std::move(device.createFramebufferUnique(fbci).value);

		  VKU_SET_OBJECT_NAME(vk::ObjectType::eFramebuffer, (VkFramebuffer)*framebuffers_[eFrameBuffers::COLOR_DEPTH][i], vkNames::FrameBuffer::COLOR_DEPTH);
	  }

	  // Final Pass to Present (Post AA)
	  {
		  vku::RenderpassMaker rpm;

		  // The colour attachment. (resolved)
		  rpm.attachmentBegin(swapchainImageFormat_);
		  rpm.attachmentSamples(vk::SampleCountFlagBits::e1);					 // 0

		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eStore);
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eUndefined);
		  rpm.attachmentFinalLayout(vk::ImageLayout::ePresentSrcKHR);		// any attachment using imageViews_[i] attachments **stay on ePresentSrcKHR explicitly hereafter**

		  // A subpass to render using the above attachment.
		  rpm.subpassBegin(vk::PipelineBindPoint::eGraphics);
		  rpm.subpassColorAttachment(vk::ImageLayout::eColorAttachmentOptimal, 0);

		  // subpass 1 - overlay final
		  // The colour attachment. (resolved)
		  rpm.attachmentBegin(swapchainImageFormat_);
		  rpm.attachmentSamples(vk::SampleCountFlagBits::e1);					// 1

		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eLoad);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eStore);
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::ePresentSrcKHR);
		  rpm.attachmentFinalLayout(vk::ImageLayout::ePresentSrcKHR);

		  // The input attachment. (gui 0)
		  rpm.attachmentBegin(guiImage_.resolved[0].format());
		  rpm.attachmentSamples(vk::SampleCountFlagBits::e1);					// 2

		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eLoad);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

		  // The input attachment. (gui 1)
		  rpm.attachmentBegin(guiImage_.resolved[1].format());
		  rpm.attachmentSamples(vk::SampleCountFlagBits::e1);					// 3

		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eLoad);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

		  // A subpass to render using the above attachment.
		  rpm.subpassBegin(vk::PipelineBindPoint::eGraphics);
		  rpm.subpassColorAttachment(vk::ImageLayout::eColorAttachmentOptimal, 1);
		  rpm.subpassInputAttachment(vk::ImageLayout::eShaderReadOnlyOptimal, 2);
		  rpm.subpassInputAttachment(vk::ImageLayout::eShaderReadOnlyOptimal, 3);

		  /* Chapter 32 of Vulkan Spec
		  When transitioning the image to VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR or VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, there is no need to delay subsequent processing, or perform any visibility operations (as vkQueuePresentKHR performs automatic visibility operations). To achieve this, the dstAccessMask member of the VkImageMemoryBarrier should be set to 0, and the dstStageMask parameter should be set to VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT.
		  */

		  rpm.dependencyBegin(VK_SUBPASS_EXTERNAL, 0);
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
		  rpm.dependencyDstAccessMask( vk::AccessFlagBits::eColorAttachmentWrite);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  rpm.dependencyBegin(VK_SUBPASS_EXTERNAL, 1);
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eFragmentShader);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eInputAttachmentRead);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  rpm.dependencyBegin(0, 1);
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eColorAttachmentRead|vk::AccessFlagBits::eColorAttachmentWrite);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  rpm.dependencyBegin(1, VK_SUBPASS_EXTERNAL); // Out To Present
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eBottomOfPipe);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eColorAttachmentRead);
		  rpm.dependencyDstAccessMask((vk::AccessFlags)0);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  // Use the maker object to construct the vulkan object
		  finalPass_ = rpm.createUnique(device);

		  VKU_SET_OBJECT_NAME(vk::ObjectType::eRenderPass, (VkRenderPass)*finalPass_, vkNames::Renderpass::FINAL);
	  }

	  framebuffers_[eFrameBuffers::PRESENT] = new vk::UniqueFramebuffer[imageViews_.size()];
	  for (int i = 0; i != imageViews_.size(); ++i) {
		  vk::ImageView const attachments[4] = { imageViews_[i], imageViews_[i], guiImage_.resolved[0].imageView(), guiImage_.resolved[1].imageView() };
		  vk::FramebufferCreateInfo const fbci{ {}, *finalPass_, _countof(attachments), attachments, width_, height_, 1 };
		  framebuffers_[eFrameBuffers::PRESENT][i] = std::move(device.createFramebufferUnique(fbci).value);

		  VKU_SET_OBJECT_NAME(vk::ObjectType::eFramebuffer, (VkFramebuffer)*framebuffers_[eFrameBuffers::PRESENT][i], vkNames::FrameBuffer::PRESENT);
	  }


	  // ###### Offscreen Special RenderPass
	  // Build the renderpass using one attachment, colour   (mid/intermediatte pass)
	  {
		  vku::RenderpassMaker rpm;

		  // *** SUBPASS 0 - Isolated regular rendering //

		  // The only colour attachment.
		  rpm.attachmentBegin(offscreenImage_.multisampled.format());							// 0
		  rpm.attachmentSamples(vku::DefaultSampleCount);
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eClear);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eUndefined);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eColorAttachmentOptimal);

		  // The depth/stencil attachment.
		  rpm.attachmentBegin(depthImage_.format());								// 1
		  rpm.attachmentSamples(vku::DefaultSampleCount);
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eLoad);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eDepthStencilReadOnlyOptimal);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eDepthStencilReadOnlyOptimal);

		  // The input attachment. (reflection)
		  rpm.attachmentBegin(colorReflectionImage_.upsampled.format());							// 2
		  rpm.attachmentSamples(vk::SampleCountFlagBits::e1);
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eLoad);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

		  // The resolved colour attachment.			// resolve for offscreen
		  rpm.attachmentBegin(offscreenImage_.resolved.format());
		  rpm.attachmentSamples(vk::SampleCountFlagBits::e1);				// 3
		  rpm.attachmentLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStoreOp(vk::AttachmentStoreOp::eStore);
		  rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
		  rpm.attachmentStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
		  rpm.attachmentInitialLayout(vk::ImageLayout::eUndefined);
		  rpm.attachmentFinalLayout(vk::ImageLayout::eShaderReadOnlyOptimal);	// allow sampling, for copy external image barriers are used only when a copy is enabled

		  // A subpass to render using the above two attachments.
		  rpm.subpassBegin(vk::PipelineBindPoint::eGraphics);
		  rpm.subpassColorAttachment(vk::ImageLayout::eColorAttachmentOptimal, 0);
		  rpm.subpassDepthStencilAttachment(vk::ImageLayout::eDepthStencilReadOnlyOptimal, 1);	// optimal format (read/write) during subpass
		  rpm.subpassInputAttachment(vk::ImageLayout::eShaderReadOnlyOptimal, 2);
		  rpm.subpassResolveAttachment(vk::ImageLayout::eColorAttachmentOptimal, 3);


		  rpm.dependencyBegin(VK_SUBPASS_EXTERNAL, 0);
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eTopOfPipe);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		  rpm.dependencySrcAccessMask((vk::AccessFlags)0);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  rpm.dependencyBegin(VK_SUBPASS_EXTERNAL, 0);
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests);
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eDepthStencilAttachmentRead);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eDepthStencilAttachmentRead);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  // reflection already in correct state (previous midpass), no dependency needed already input attachment - shaderreadonly

		  rpm.dependencyBegin(0, VK_SUBPASS_EXTERNAL);
		  rpm.dependencySrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);  // offscreen available for sampling
		  rpm.dependencyDstStageMask(vk::PipelineStageFlagBits::eFragmentShader);
		  rpm.dependencySrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
		  rpm.dependencyDstAccessMask(vk::AccessFlagBits::eShaderRead);
		  rpm.dependencyDependencyFlags(vk::DependencyFlagBits::eByRegion);

		  // Use the maker object to construct the vulkan object
		  offscreenPass_ = rpm.createUnique(device);

		  VKU_SET_OBJECT_NAME(vk::ObjectType::eRenderPass, (VkRenderPass)*offscreenPass_, vkNames::Renderpass::OFFSCREEN);
	  }

	  framebuffers_[eFrameBuffers::OFFSCREEN] = new vk::UniqueFramebuffer[imageViews_.size()];
	  for (int i = 0; i != imageViews_.size(); ++i) {
		  vk::ImageView const attachments[4] = { offscreenImage_.multisampled.imageView(), depthImage_.imageView(), colorReflectionImage_.upsampled.imageView(), offscreenImage_.resolved.imageView() };
		  vk::FramebufferCreateInfo const fbci{ {}, *offscreenPass_, _countof(attachments), attachments, width_, height_, 1 };
		  framebuffers_[eFrameBuffers::OFFSCREEN][i] = std::move(device.createFramebufferUnique(fbci).value);

		  VKU_SET_OBJECT_NAME(vk::ObjectType::eFramebuffer, (VkFramebuffer)*framebuffers_[eFrameBuffers::OFFSCREEN][i], vkNames::FrameBuffer::OFFSCREEN);
	  }

	  {
		  vk::SemaphoreCreateInfo sci;
		  for (uint32_t i = 0; i < 2; ++i) {
			  semaphores[i].imageAcquireSemaphore_ = device.createSemaphoreUnique(sci).value;
			  semaphores[i].commandCompleteSemaphore_ = device.createSemaphoreUnique(sci).value;
			  semaphores[i].transferCompleteSemaphore_[0] = device.createSemaphoreUnique(sci).value;	// compute transfer
			  semaphores[i].transferCompleteSemaphore_[1] = device.createSemaphoreUnique(sci).value;	// dynamic transfer
			  semaphores[i].computeCompleteSemaphore_[0] = device.createSemaphoreUnique(sci).value;		// compute process light
			  semaphores[i].computeCompleteSemaphore_[1] = device.createSemaphoreUnique(sci).value;		// compute process textures
		  }
	  }
	  
	  typedef vk::CommandPoolCreateFlagBits ccbits;

	  {
		  vk::CommandPoolCreateInfo cpci{ ccbits::eTransient | ccbits::eResetCommandBuffer, graphicsQueueFamilyIndex };
		  commandPool_[eCommandPools::DEFAULT_POOL] = device.createCommandPoolUnique(cpci).value;
	  }
	  {
		  vk::CommandPoolCreateInfo cpci{ ccbits::eTransient | ccbits::eResetCommandBuffer, graphicsQueueFamilyIndex };
		  commandPool_[eCommandPools::OVERLAY_POOL] = device.createCommandPoolUnique(cpci).value;
	  }
	  {

		  vk::CommandPoolCreateInfo cpci{ ccbits::eTransient | ccbits::eResetCommandBuffer, transferQueueFamilyIndex };
		  commandPool_[eCommandPools::DMA_TRANSFER_POOL_PRIMARY] = device.createCommandPoolUnique(cpci).value;
		  commandPool_[eCommandPools::DMA_TRANSFER_POOL_SECONDARY] = device.createCommandPoolUnique(cpci).value;
	  }
	  {
		  vk::CommandPoolCreateInfo cpci{ ccbits::eTransient | ccbits::eResetCommandBuffer, computeQueueFamilyIndex };
		  commandPool_[eCommandPools::COMPUTE_POOL_PRIMARY] = device.createCommandPoolUnique(cpci).value;
		  commandPool_[eCommandPools::COMPUTE_POOL_SECONDARY] = device.createCommandPoolUnique(cpci).value;
	  }

	  // Create draw buffers
	  { // static
		  uint32_t const resource_count((uint32_t)imageViews_.size());
		  vk::CommandBufferAllocateInfo cbai{ *commandPool_[eCommandPools::DEFAULT_POOL], vk::CommandBufferLevel::ePrimary, resource_count };
		  staticDrawBuffers_.allocate(device, cbai);
		 
		  for (uint32_t i = 0; i < resource_count; ++i) {
			  staticCommandsDirty_.emplace_back(false);
			  VKU_SET_OBJECT_NAME(vk::ObjectType::eCommandBuffer, (VkCommandBuffer)*staticDrawBuffers_.cb[0][i], vkNames::CommandBuffer::STATIC);
		  }
	  }
	  {	// present command buffer is fully static and cannot be changed - no diry flag
		  uint32_t const resource_count((uint32_t)imageViews_.size());
		  vk::CommandBufferAllocateInfo cbai{ *commandPool_[eCommandPools::DEFAULT_POOL], vk::CommandBufferLevel::ePrimary, resource_count };
		  presentDrawBuffers_.allocate(device, cbai);
		  for (uint32_t i = 0; i < resource_count; ++i) {
			  VKU_SET_OBJECT_NAME(vk::ObjectType::eCommandBuffer, (VkCommandBuffer)* presentDrawBuffers_.cb[0][i], vkNames::CommandBuffer::PRESENT);
		  }
	  }
	  { // overlay render
		  uint32_t const resource_count((uint32_t)imageViews_.size());
		  vk::CommandBufferAllocateInfo cbai{ *commandPool_[eCommandPools::OVERLAY_POOL], vk::CommandBufferLevel::ePrimary, resource_count };
		  overlayDrawBuffers_.allocate<eOverlayBuffers::RENDER>(device, cbai);
		  for (uint32_t i = 0; i < resource_count; ++i) {
			  VKU_SET_OBJECT_NAME(vk::ObjectType::eCommandBuffer, (VkCommandBuffer)*overlayDrawBuffers_.cb[eOverlayBuffers::RENDER][i], vkNames::CommandBuffer::OVERLAY_RENDER);
		  }
	  }

	  {
		  { // dynamic
			  uint32_t const resource_count((uint32_t)imageViews_.size());
			  vk::CommandBufferAllocateInfo cbai{ *commandPool_[eCommandPools::DMA_TRANSFER_POOL_PRIMARY], vk::CommandBufferLevel::ePrimary, resource_count };
			  dynamicDrawBuffers_.allocate(device, cbai);
			  for (uint32_t i = 0; i < resource_count; ++i) {
				  VKU_SET_OBJECT_NAME(vk::ObjectType::eCommandBuffer, (VkCommandBuffer)* dynamicDrawBuffers_.cb[0][i], vkNames::CommandBuffer::DYNAMIC);
			  }
		  }
		  { // compute transfer
			  uint32_t const resource_count(2U);
			  vk::CommandBufferAllocateInfo cbai{ *commandPool_[eCommandPools::DMA_TRANSFER_POOL_PRIMARY], vk::CommandBufferLevel::ePrimary, resource_count }; // 2 resources
			  computeDrawBuffers_.allocate<eComputeBuffers::TRANSFER_LIGHT>(device, cbai);
			  for (uint32_t i = 0; i < resource_count; ++i) {
				  computeCommandsDirty_[i] = false;
				  VKU_SET_OBJECT_NAME(vk::ObjectType::eCommandBuffer, (VkCommandBuffer)* computeDrawBuffers_.cb[eComputeBuffers::TRANSFER_LIGHT][i], vkNames::CommandBuffer::TRANSFER_LIGHT);
			  }
		  }
		  { // overlay transfer
			  uint32_t const resource_count((uint32_t)imageViews_.size());
			  vk::CommandBufferAllocateInfo cbai{ *commandPool_[eCommandPools::DMA_TRANSFER_POOL_SECONDARY], vk::CommandBufferLevel::ePrimary, resource_count };
			  overlayDrawBuffers_.allocate<eOverlayBuffers::TRANSFER>(device, cbai);
			  for (uint32_t i = 0; i < resource_count; ++i) {
				  VKU_SET_OBJECT_NAME(vk::ObjectType::eCommandBuffer, (VkCommandBuffer)*overlayDrawBuffers_.cb[eOverlayBuffers::TRANSFER][i], vkNames::CommandBuffer::OVERLAY_TRANSFER);
			  }
		  }
	  }
	  { // compute render
		  uint32_t const resource_count(2U);
		  vk::CommandBufferAllocateInfo cbai{ *commandPool_[eCommandPools::COMPUTE_POOL_PRIMARY], vk::CommandBufferLevel::ePrimary, resource_count };	// 2 resources
		  computeDrawBuffers_.allocate<eComputeBuffers::COMPUTE_LIGHT>(device, cbai);
		  for (uint32_t i = 0; i < resource_count; ++i) {
			  VKU_SET_OBJECT_NAME(vk::ObjectType::eCommandBuffer, (VkCommandBuffer)* computeDrawBuffers_.cb[eComputeBuffers::COMPUTE_LIGHT][i], vkNames::CommandBuffer::COMPUTE_LIGHT);
		  }
	  }
	  { // compute textureShaders
		  uint32_t const resource_count(2U);
		  vk::CommandBufferAllocateInfo cbai{ *commandPool_[eCommandPools::COMPUTE_POOL_SECONDARY], vk::CommandBufferLevel::ePrimary, resource_count };	// 2 resources
		  computeDrawBuffers_.allocate<eComputeBuffers::COMPUTE_TEXTURE>(device, cbai);
		  for (uint32_t i = 0; i < resource_count; ++i) {
			  VKU_SET_OBJECT_NAME(vk::ObjectType::eCommandBuffer, (VkCommandBuffer)*computeDrawBuffers_.cb[eComputeBuffers::COMPUTE_TEXTURE][i], vkNames::CommandBuffer::COMPUTE_TEXTURE);
		  }
	  }

	  initializeCheckerboardStencilBufferImages(*commandPool_[eCommandPools::TRANSIENT_POOL], graphicsQueue_);

    ok_ = true; 
  }

  /// Dump the capabilities of the physical device used by this window.
  /*
  void dumpCaps(std::ostream &os, vk::PhysicalDevice pd) const {
    os << "Surface formats\n";
    auto fmts = pd.getSurfaceFormatsKHR(surface_).value;
    for (auto &fmt : fmts) {
      auto fmtstr = vk::to_string(fmt.format);
      auto cstr = vk::to_string(fmt.colorSpace);
      os << "format=" << fmtstr << " colorSpace=" << cstr << "\n";
    }

    os << "Present Modes\n";
    auto presentModes = pd.getSurfacePresentModesKHR(surface_).value;
    for (auto pm : presentModes) {
      std::cout << vk::to_string(pm) << "\n";
    }
  }
  */

  static void defaultRenderFunc(vk::CommandBuffer cb, int imageIndex, vk::RenderPassBeginInfo const&rpbi) {
    vk::CommandBufferBeginInfo bi{};
    cb.begin(bi);
    cb.end();
  }

  static inline static_renderpass_function_unconst staticCommandCache;

  /// Build a static draw buffer. This will be rendered after any dynamic content generated in draw()
  void setStaticCommands(static_renderpass_function static_function, int32_t const iImageIndex = -1) {

	  // alpha channel ust atleast be cleared to 1 for transparency "clear masks"
	  // it is faster to clear all channels to 1 or 0
	  static vk::ClearValue const clearArray_zPass[] = { vk::ClearDepthStencilValue{1.0f, 0}, vk::ClearColorValue{ std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f}}, vk::ClearColorValue{ std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f}},  {}, {}, {}, {} };
	  static vk::ClearValue const clear_offscreenPass{ vk::ClearColorValue{ std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}} };  // require opaque alpha, no alpha component writes in voxel shader due to clear masks

	  point2D const frameBufferSz(width_, height_);
	  point2D_t const downResFrameBufferSz(vku::getDownResolution(frameBufferSz));

	  vk::RenderPassBeginInfo rpbi[5];

	  rpbi[eFrameBuffers::DEPTH].renderPass = *zPass_;
	  rpbi[eFrameBuffers::DEPTH].renderArea = vk::Rect2D{ {0, 0}, {width_, height_} };
	  rpbi[eFrameBuffers::DEPTH].clearValueCount = (uint32_t)_countof(clearArray_zPass);
	  rpbi[eFrameBuffers::DEPTH].pClearValues = clearArray_zPass;

	  rpbi[eFrameBuffers::HALF_COLOR_ONLY].renderPass = *downPass_;
	  rpbi[eFrameBuffers::HALF_COLOR_ONLY].renderArea = vk::Rect2D{ {0, 0}, {uint32_t(downResFrameBufferSz.x), uint32_t(downResFrameBufferSz.y)} };
	  rpbi[eFrameBuffers::HALF_COLOR_ONLY].clearValueCount = 0;;
	  rpbi[eFrameBuffers::HALF_COLOR_ONLY].pClearValues = nullptr;

	  rpbi[eFrameBuffers::FULL_COLOR_ONLY].renderPass = *upPass_;
	  rpbi[eFrameBuffers::FULL_COLOR_ONLY].renderArea = vk::Rect2D{ {0, 0}, {width_, height_} };
	  rpbi[eFrameBuffers::FULL_COLOR_ONLY].clearValueCount = 0;;
	  rpbi[eFrameBuffers::FULL_COLOR_ONLY].pClearValues = nullptr;

	  rpbi[eFrameBuffers::MID_COLOR_DEPTH].renderPass = *midPass_;
	  rpbi[eFrameBuffers::MID_COLOR_DEPTH].renderArea = vk::Rect2D{ {0, 0}, {width_, height_} };
	  rpbi[eFrameBuffers::MID_COLOR_DEPTH].clearValueCount = 0;
	  rpbi[eFrameBuffers::MID_COLOR_DEPTH].pClearValues = nullptr;

	  static constexpr uint32_t const OFFSCREEN_OFFSET(eFrameBuffers::OFFSCREEN - 2);
	  rpbi[OFFSCREEN_OFFSET].renderPass = *offscreenPass_;
	  rpbi[OFFSCREEN_OFFSET].renderArea = vk::Rect2D{ {0, 0}, {width_, height_} };
	  rpbi[OFFSCREEN_OFFSET].clearValueCount = 1;
	  rpbi[OFFSCREEN_OFFSET].pClearValues = &clear_offscreenPass;

	  if (iImageIndex < 0) {
		  for (int i = 0; i != staticDrawBuffers_.size(); ++i) {
			  vk::CommandBuffer const cb = *staticDrawBuffers_.cb[0][i];
			  rpbi[eFrameBuffers::DEPTH].framebuffer			= *framebuffers_[eFrameBuffers::DEPTH][i];
			  rpbi[eFrameBuffers::HALF_COLOR_ONLY].framebuffer  = *framebuffers_[eFrameBuffers::HALF_COLOR_ONLY][i];
			  rpbi[eFrameBuffers::FULL_COLOR_ONLY].framebuffer  = *framebuffers_[eFrameBuffers::FULL_COLOR_ONLY][i];
			  rpbi[eFrameBuffers::MID_COLOR_DEPTH].framebuffer  = *framebuffers_[eFrameBuffers::MID_COLOR_DEPTH][i];
			  rpbi[OFFSCREEN_OFFSET].framebuffer				= *framebuffers_[eFrameBuffers::OFFSCREEN][i];

			  static_function(std::forward<static_renderpass&& __restrict>({ cb, 
				  std::move(rpbi[eFrameBuffers::DEPTH]), 
				  std::move(rpbi[eFrameBuffers::HALF_COLOR_ONLY]), 
				  std::move(rpbi[eFrameBuffers::FULL_COLOR_ONLY]), 
				  std::move(rpbi[eFrameBuffers::MID_COLOR_DEPTH]), 
				  std::move(rpbi[OFFSCREEN_OFFSET]) }));

			  staticCommandsDirty_[i] = false;
		  }
		  
		  staticCommandCache = static_function;
	  }
	  else {
		  vk::CommandBuffer const cb = *staticDrawBuffers_.cb[0][iImageIndex];
		  rpbi[eFrameBuffers::DEPTH].framebuffer			= *framebuffers_[eFrameBuffers::DEPTH][iImageIndex];
		  rpbi[eFrameBuffers::HALF_COLOR_ONLY].framebuffer  = *framebuffers_[eFrameBuffers::HALF_COLOR_ONLY][iImageIndex];
		  rpbi[eFrameBuffers::FULL_COLOR_ONLY].framebuffer  = *framebuffers_[eFrameBuffers::FULL_COLOR_ONLY][iImageIndex];
		  rpbi[eFrameBuffers::MID_COLOR_DEPTH].framebuffer  = *framebuffers_[eFrameBuffers::MID_COLOR_DEPTH][iImageIndex];
		  rpbi[OFFSCREEN_OFFSET].framebuffer				= *framebuffers_[eFrameBuffers::OFFSCREEN][iImageIndex];

		  static_function(std::forward<static_renderpass&& __restrict>({ cb, 
			  std::move(rpbi[eFrameBuffers::DEPTH]), 
			  std::move(rpbi[eFrameBuffers::HALF_COLOR_ONLY]), 
			  std::move(rpbi[eFrameBuffers::FULL_COLOR_ONLY]), 
			  std::move(rpbi[eFrameBuffers::MID_COLOR_DEPTH]), 
			  std::move(rpbi[OFFSCREEN_OFFSET]) }));

		  staticCommandsDirty_[iImageIndex] = false;
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

  /// Build a static draw buffer. This will be rendered after any dynamic content generated in draw()
  void setStaticPresentCommands(present_renderpass_function present_function) {

	  vk::RenderPassBeginInfo rpbi;
	  rpbi.renderPass = *finalPass_;
	  rpbi.renderArea = vk::Rect2D{ {0, 0}, {width_, height_} };
	  rpbi.clearValueCount = 0U;
	  rpbi.pClearValues = nullptr;

		for (uint32_t i = 0; i != presentDrawBuffers_.size(); ++i) {
			vk::CommandBuffer const cb = *presentDrawBuffers_.cb[0][i];
			rpbi.framebuffer = *framebuffers_[eFrameBuffers::PRESENT][i];

			present_function(std::forward<present_renderpass&& __restrict>({ cb, std::move(rpbi), i }));
		}
  }
  
#ifdef VKU_IMPLEMENTATION

  private:
	  NO_INLINE bool const fail_acquire_or_present(vk::Result const result, uint32_t& imageIndex, uint32_t& resource_index)
	  {
		  bool bReturn(false);

		  switch (result)
		  {
		  case vk::Result::eSuccess: // should never get here but if we do silently ignore
			  break;
		  case vk::Result::eSuboptimalKHR:	// silently recreate the swap chain, then pre-acquire first image, reset image and resource indices
		  case vk::Result::eErrorOutOfDateKHR:
		  {
			  ClipCursor(nullptr);  // bugfix, release cursor //
			  device_.waitIdle();	// safetly continue after idle detect
			  recreateSwapChain();

			  imageIndex = 0;
			  resource_index = 0;
			  bReturn = true;
		  }
		  break;
		  default:
			  ClipCursor(nullptr);  // bugfix, release cursor //
			  FMT_LOG_FAIL(GPU_LOG, "Major failure in main render method, < {:s} > ", vk::to_string(result));
			  break;
		  };

		  return(bReturn); // indicating current render frame should be aborted and re attempted next frame
	  }
	  
  public:
  /// Queue the static command buffer for the next image in the swap chain. Optionally call a function to create a dynamic command buffer
  /// for uploading textures, changing uniforms etc.
  void draw(const vk::Device& __restrict device,
	  compute_function gpu_compute, dynamic_renderpass_function dynamic_function, overlay_renderpass_function overlay_function) {

	  // [ RENDERGRAPH ]-------------------------------------------------------------------------------------------------------------
	  //  
	  // 	   
	  // [ COMPUTE (TEXTURESHADERS) ] -------------------------------------| 
	  // [ UPLOAD (LIGHT) ] ---------------[ COMPUTE (LIGHT) ]----------------[ STATIC ]-----[ OVERLAY ]-----[ POST & PRESENT ]
	  // [ UPLOAD (DYNAMIC + OVERLAY) ] -----------------------------------|
	  //
	  // 

	  // utilize the time between a present() and acquireNextImage()
	  static constexpr uint64_t const umax = nanoseconds(milliseconds(500)).count();

	  static uint32_t
		  resource_index{};		// **** only "compute, dynamic, post_submit_render" should use the resource_index, otherwise use imageIndex ******
		  						// dynamic uses imageIndex, but uses resource_index to refer to the objects worked on in post_submit_render
	  
	  resource_control::stage_resources(resource_index);				

	  // ###################################################### //
	  async_long_task::wait<background_critical>(present_task_id, "present");  // bugfix: absolutely critical that this is done here, any other location results in validation THREADING errors
	  // ####################################################### //
	  
	  int64_t
		  task_compute_textures(0),
		  task_compute_light(0);

	  // COMPUTE SUBMIT (TEXTURESHADERS)// // **waiting on nothing
	  vk::Semaphore const ctexSema = { *semaphores[resource_index].computeCompleteSemaphore_[eComputeBuffers::COMPUTE_TEXTURE - 1] };
	  {
		  task_compute_textures = async_long_task::enqueue<background_critical>(
			  // non-blocking submit
			  [=] {
				  vk::CommandBuffer const compute_process[3] = { nullptr, nullptr, *computeDrawBuffers_.cb[eComputeBuffers::COMPUTE_TEXTURE][resource_index] };

				  vk::Fence const compute_fence = computeDrawBuffers_.fence[eComputeBuffers::COMPUTE_TEXTURE][resource_index];
				  if (computeDrawBuffers_.queued[eComputeBuffers::COMPUTE_TEXTURE][resource_index]) {
					  device.waitForFences(compute_fence, VK_TRUE, umax);
					  device.resetFences(compute_fence);
					  computeDrawBuffers_.queued[eComputeBuffers::COMPUTE_TEXTURE][resource_index] = false; // reset
				  }

				  gpu_compute(std::forward<compute_pass&& __restrict>({ compute_process[eComputeBuffers::TRANSFER_LIGHT], compute_process[eComputeBuffers::COMPUTE_LIGHT], compute_process[eComputeBuffers::COMPUTE_TEXTURE], resource_index }));    // compute part resets the dirty state that transfer set

				  //vk::PipelineStageFlags waitStages{ vk::PipelineStageFlagBits::eComputeShader };
				  vk::SubmitInfo submit{};
				  submit.waitSemaphoreCount = 0;
				  submit.pWaitSemaphores = nullptr;				// **waiting on nothing
				  submit.pWaitDstStageMask = nullptr;
				  submit.commandBufferCount = 1;
				  submit.pCommandBuffers = &compute_process[eComputeBuffers::COMPUTE_TEXTURE];				// submitting compute cb
				  submit.signalSemaphoreCount = 1;
				  submit.pSignalSemaphores = &ctexSema;			// signalling compute cb completion

				  computeQueue_[!resource_index].submit(1, &submit, compute_fence); // always use "other" compute queue so they potentially can be running independently and in parallel

				  computeDrawBuffers_.queued[eComputeBuffers::COMPUTE_TEXTURE][resource_index] = true;

			  });
	  }

	  // UPLOAD (LIGHT) // // **waiting on nothing
	  vk::Semaphore const tcSema[2] = { *semaphores[resource_index].transferCompleteSemaphore_[0], *semaphores[resource_index].transferCompleteSemaphore_[1] };

	  {
		vk::Fence const dma_transfer_light_fence = computeDrawBuffers_.fence[eComputeBuffers::TRANSFER_LIGHT][resource_index];
		if (computeDrawBuffers_.queued[eComputeBuffers::TRANSFER_LIGHT][resource_index]) {
			device.waitForFences(dma_transfer_light_fence, VK_TRUE, umax);			// protect
			device.resetFences(dma_transfer_light_fence);
			computeDrawBuffers_.queued[eComputeBuffers::TRANSFER_LIGHT][resource_index] = false; // reset
		}

		vk::CommandBuffer const compute_upload_light[3] = { *computeDrawBuffers_.cb[eComputeBuffers::TRANSFER_LIGHT][resource_index], nullptr, nullptr };

		if (!computeCommandsDirty_[resource_index])  // only if not dirty already
		{
			// upload light
			computeCommandsDirty_[resource_index] = gpu_compute(std::forward<compute_pass&& __restrict>({ compute_upload_light[eComputeBuffers::TRANSFER_LIGHT], compute_upload_light[eComputeBuffers::COMPUTE_LIGHT], compute_upload_light[eComputeBuffers::COMPUTE_TEXTURE], resource_index }));
		}

		// COMPUTE DMA TRANSFER SUBMIT //
		if (computeCommandsDirty_[resource_index]) {									// seperated queue sb,it for transfering light and opacity
																						// so compute only waits on transfer light semaphore, and static waits on opacity semaphore
			vk::SubmitInfo submit{};																			// rather than compute waiting on combined transfer of both light + opacity. Compute only dependent on light.
			submit.waitSemaphoreCount = 0;
			submit.pWaitSemaphores = nullptr;			// **waiting on nothing
			submit.pWaitDstStageMask = nullptr;
			submit.commandBufferCount = 1;				// submitting dma cb
			submit.pCommandBuffers = &compute_upload_light[eComputeBuffers::TRANSFER_LIGHT];
			submit.signalSemaphoreCount = 1;
			submit.pSignalSemaphores = &tcSema[0];			// signal for compute

			transferQueue_[resource_index].submit(1, &submit, dma_transfer_light_fence);

			computeDrawBuffers_.queued[eComputeBuffers::TRANSFER_LIGHT][resource_index] = true;
		}
	  } 

	  // UPLOAD & OVERLAY // // **waiting on nothing
	  vk::Fence const overlay_fence = overlayDrawBuffers_.fence[eOverlayBuffers::TRANSFER][0];   /// <---- this is zero on purpose rather than resource index
	  {																							 //  bugfix for flickering of voxels & gui!
		vk::CommandBuffer do_cb[2] = { *dynamicDrawBuffers_.cb[0][resource_index], *overlayDrawBuffers_.cb[eOverlayBuffers::TRANSFER][resource_index] };

		vk::Fence const dynamic_fence = dynamicDrawBuffers_.fence[0][resource_index];

		{ // ######### begin overlay transfer cb update (spawned)
			device.waitForFences(overlay_fence, VK_TRUE, umax);				// protect // overlay fence is overlay Render cb
			device.resetFences(overlay_fence);

			// staging
			overlay_function(std::forward<overlay_renderpass&& __restrict>({ &do_cb[1], nullptr, std::forward<vk::RenderPassBeginInfo&& __restrict>(vk::RenderPassBeginInfo{}) }));		// submission of staged data to gpu // build transfer cb
		}

		{ // ######### begin dynamic transfer cb update (main thread)
			device.waitForFences(dynamic_fence, VK_TRUE, umax);		// protect
			device.resetFences(dynamic_fence);

			dynamic_function(std::forward<dynamic_renderpass&&>({ do_cb[0], resource_index }));	// submission of staged data to gpu
		}

		// DYNAMIC & OVERLAY DYNAMIC SUBMIT //
		{
			vk::SubmitInfo submit{};
			submit.waitSemaphoreCount = 0;
			submit.pWaitSemaphores = nullptr;				// **waiting on nothing
			submit.pWaitDstStageMask = nullptr;
			submit.commandBufferCount = 2;				// submitting dynamic cb & overlay's dynamic cb
			submit.pCommandBuffers = do_cb;
			submit.signalSemaphoreCount = 1;
			submit.pSignalSemaphores = &tcSema[1];			// signal for dynamic cb in slot 0, signal for overlay dynamic cb in slot 1 (completion)

			transferQueue_[!resource_index].submit(1, &submit, dynamic_fence); // <---- this is opposite transfer queue on purpose so dma transfers are simultaneous
		}
	  }

	  // COMPUTE SUBMIT // // **waiting on upload light
	  vk::Semaphore const cSema = { *semaphores[resource_index].computeCompleteSemaphore_[eComputeBuffers::COMPUTE_LIGHT - 1] };
	  bool bAsyncCompute(false);

	  if (computeCommandsDirty_[resource_index]) {

		  task_compute_light = async_long_task::enqueue<background_critical>(
			  // non-blocking submit
			  [=, &bAsyncCompute] {
				  vk::CommandBuffer const compute_process[3] = { nullptr, *computeDrawBuffers_.cb[eComputeBuffers::COMPUTE_LIGHT][resource_index], nullptr };

				  vk::Fence const compute_fence = computeDrawBuffers_.fence[eComputeBuffers::COMPUTE_LIGHT][resource_index];
				  if (computeDrawBuffers_.queued[eComputeBuffers::COMPUTE_LIGHT][resource_index]) {
					  device.waitForFences(compute_fence, VK_TRUE, umax);
					  device.resetFences(compute_fence);
					  computeDrawBuffers_.queued[eComputeBuffers::COMPUTE_LIGHT][resource_index] = false; // reset
				  }

				  computeCommandsDirty_[resource_index] = gpu_compute(std::forward<compute_pass&& __restrict>({ compute_process[eComputeBuffers::TRANSFER_LIGHT], compute_process[eComputeBuffers::COMPUTE_LIGHT], compute_process[eComputeBuffers::COMPUTE_TEXTURE], resource_index }));    // compute part resets the dirty state that transfer set

				  vk::PipelineStageFlags waitStages{ vk::PipelineStageFlagBits::eComputeShader };

				  vk::SubmitInfo submit{};
				  submit.waitSemaphoreCount = (uint32_t)computeDrawBuffers_.queued[eComputeBuffers::TRANSFER_LIGHT][resource_index];
				  submit.pWaitSemaphores = &tcSema[0];				// waiting on transfer completion only if transfer in progress, otherwise waiting on nothing
				  submit.pWaitDstStageMask = &waitStages;
				  submit.commandBufferCount = 1;
				  submit.pCommandBuffers = &compute_process[eComputeBuffers::COMPUTE_LIGHT];				// submitting compute cb
				  submit.signalSemaphoreCount = 1;
				  submit.pSignalSemaphores = &cSema;			// signalling compute cb completion
				  computeQueue_[resource_index].submit(1, &submit, compute_fence);

				  computeDrawBuffers_.queued[eComputeBuffers::COMPUTE_LIGHT][resource_index] = true;
				  bAsyncCompute = true;
			  });
	  }

	  // upload & compute
	  // 	   |
	  // 	graphics
	  
	  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
	  // ANY WORK THAT CAN BE DONE (COMPUTE, TRANSFERS, ANYTHING THAT DOES NOT DEPEND ON IMAGEINDEX) SHOULD BE DONE ABOVE //
	  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

	  vk::Result result(vk::Result::eSuccess);

	  uint32_t imageIndex{};
	  vk::Semaphore const iaSema = *semaphores[resource_index].imageAcquireSemaphore_;

	  result = device.acquireNextImageKHR(*swapchain_, umax, iaSema, vk::Fence(), &imageIndex);	// **** driver does all of its waiting / spinning here blocking any further execution until ready!!!
																												  // do any / all updates before this call to spend the time wisely
	  [[unlikely]] if (vk::Result::eSuccess != result || vk::Result::eSuccess != _presentResult) {
		  if ( fail_acquire_or_present(result, imageIndex, resource_index) )
			return;
	  }
	  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

    async_long_task::wait<background_critical>(task_compute_textures, "compute textureshaders");
    async_long_task::wait<background_critical>(task_compute_light, "compute light");

	vk::Semaphore const ccSema = *semaphores[resource_index].commandCompleteSemaphore_;

	int64_t const task_graphics = async_long_task::enqueue<background_critical>(
		// non-blocking submit
		[=] {
			vk::Semaphore const iatexctccSema[4] = { iaSema, ctexSema, tcSema[1], cSema };

			// STATIC SUBMIT // // **waiting on input acquire, textureshaders, upload & overlay, compute light
			{
				vk::Fence const static_fence = staticDrawBuffers_.fence[0][imageIndex];

				device.waitForFences(static_fence, VK_TRUE, umax);
				device.resetFences(static_fence);
				if (staticCommandsDirty_[imageIndex]) {
					setStaticCommands(staticCommandCache, imageIndex);
				}

				vk::CommandBuffer const cb = *staticDrawBuffers_.cb[0][imageIndex];
				vk::PipelineStageFlags waitStages[4] = { vk::PipelineStageFlagBits::eVertexInput, vk::PipelineStageFlagBits::eVertexInput, vk::PipelineStageFlagBits::eFragmentShader, vk::PipelineStageFlagBits::eFragmentShader }; // wait at stage data is required

				vk::SubmitInfo submit{};
				submit.waitSemaphoreCount = 3 + (uint32_t)bAsyncCompute;
				submit.pWaitSemaphores = iatexctccSema;		// waiting on dynamic transfer & input acquire & compute processing (both texture and light)
				submit.pWaitDstStageMask = waitStages;
				submit.commandBufferCount = 1;
				submit.pCommandBuffers = &cb;				// submitting static cb
				submit.signalSemaphoreCount = 0;
				submit.pSignalSemaphores = nullptr;			// signalling static cb completion

				// ########### FRAMES FIRST USAGE OF GRAPHICS QUEUE ################ //
				graphicsQueue_.submit(1, &submit, static_fence);
			}

			//	graphics
			// 	   |
			// 	graphics

			// **inherent wait between graphics queue operations. they serialize and it is not neccessary to signal a semaphore as there is no inter-queue dependencies.

			// OVERLAY STATIC SUBMIT // **waiting on overlay upload is not necessary as the wait has already taken place in static. The semaphore combines dynamic upload + overlay upload. Static depends on dynamic uploads completion. Single semaphore. Single signal & wait finished in STATIC.
			{
				vk::CommandBuffer ob = *overlayDrawBuffers_.cb[eOverlayBuffers::RENDER][imageIndex];

				// fence not required ....
				static vk::ClearValue const clearArray[] = { {}, {}, {}, {}, vk::ClearValue{ std::array<uint32_t, 4>{0, 0, 0, 0}}, {} };
				overlay_function(overlay_renderpass{ nullptr, &ob,
					std::forward<vk::RenderPassBeginInfo&& __restrict>(vk::RenderPassBeginInfo(*overlayPass_, *framebuffers_[eFrameBuffers::COLOR_DEPTH][imageIndex], vk::Rect2D{ {0, 0}, {width_, height_} }, _countof(clearArray), clearArray)) });  // build render cb

				vk::SubmitInfo submit{};
				submit.waitSemaphoreCount = 0;
				submit.pWaitSemaphores = nullptr;			// prior submit already waited on &tcSema[1] (contains semaphor that represents dynamic + overlay transfer)
				submit.pWaitDstStageMask = nullptr;
				submit.commandBufferCount = 1;
				submit.pCommandBuffers = &ob;				// submitting overlay's static cb
				submit.signalSemaphoreCount = 0;
				submit.pSignalSemaphores = nullptr;			// signalling commands complete
				graphicsQueue_.submit(1, &submit, overlay_fence);	// ***overlay fence is reset here ok
			}

			//	graphics
			// 	   |
			// 	graphics

			// PRESENT (POST AA) FINAL SUBMIT // **waiting on nothing
			{
				vk::Fence const cbFencePresent = presentDrawBuffers_.fence[0][imageIndex];
				device.waitForFences(cbFencePresent, VK_TRUE, umax);
				device.resetFences(cbFencePresent);				// have to wait on associatted fence, and reset for next iteration

				vk::CommandBuffer const pb = *presentDrawBuffers_.cb[0][imageIndex];

				vk::SubmitInfo submit{};
				submit.waitSemaphoreCount = 0;
				submit.pWaitSemaphores = nullptr;
				submit.pWaitDstStageMask = nullptr;
				submit.commandBufferCount = 1;
				submit.pCommandBuffers = &pb;				// submitting presents' static cb
				submit.signalSemaphoreCount = 1;
				submit.pSignalSemaphores = &ccSema;			// signalling commands complete
				graphicsQueue_.submit(1, &submit, cbFencePresent);
			}
		});

	async_long_task::wait<background_critical>(task_graphics, "graphics");

	present_task_id = async_long_task::enqueue<background_critical>(
	// non-blocking present
		[this, imageIndex, ccSema] {
			// ######## Present *currentframe* //
			vk::PresentInfoKHR presentInfo;
			vk::SwapchainKHR swapchain = *swapchain_;
			presentInfo.pSwapchains = &swapchain;
			presentInfo.swapchainCount = 1;
			presentInfo.pImageIndices = &imageIndex;
			presentInfo.waitSemaphoreCount = 1;
			presentInfo.pWaitSemaphores = &ccSema;		// waiting on completion 
			_presentResult = graphicsQueue_.presentKHR(presentInfo);		// submit/present to screen queue
		});

	// swapping resources
	resource_index = !resource_index;
  }

#endif

  /// Return true if this window was created sucessfully.
  bool ok() const { return ok_; }

  /// Return the renderpass used by this window.
  vk::RenderPass const& __restrict zPass() const { return(*zPass_); }
  vk::RenderPass const& __restrict downPass() const { return(*downPass_); }
  vk::RenderPass const& __restrict upPass() const { return(*upPass_); }
  vk::RenderPass const& __restrict midPass() const { return(*midPass_); }
  vk::RenderPass const& __restrict overlayPass() const { return(*overlayPass_); }
  vk::RenderPass const& __restrict finalPass() const { return(*finalPass_); }
  vk::RenderPass const& __restrict offscreenPass() const { return(*offscreenPass_); }

  /// Destroy resources when shutting down.
  ~Window() {

#if defined(FULLSCREEN_EXCLUSIVE) && defined(VK_EXT_full_screen_exclusive)
	  if (bFullScreenExclusiveOn) {
		  vkReleaseFullScreenExclusiveModeEXT(device_, *swapchain_);
	  }
#endif

	for (uint32_t fb = 0; fb < eFrameBuffers::_size(); ++fb) {
		SAFE_DELETE_ARRAY(framebuffers_[fb]);
	}
	zPass_.release();
	downPass_.release();
	upPass_.release();
	midPass_.release();
	overlayPass_.release();
	finalPass_.release();
	offscreenPass_.release();

	for (auto& iv : imageViews_) {
		device_.destroyImageView(iv);
	}

	computeDrawBuffers_.release(device_);
	staticDrawBuffers_.release(device_);
	dynamicDrawBuffers_.release(device_);
	overlayDrawBuffers_.release(device_);

    swapchain_ = vk::UniqueSwapchainKHR{};
  }

  Window &operator=(Window &&rhs) = default;

  /// Return the width of the display.
  uint32_t width() const { return width_; }

  /// Return the height of the display.
  uint32_t height() const { return height_; }

  // queues //
  vk::Queue const& __restrict graphicsQueue() const { return(graphicsQueue_); }
  vk::Queue const& __restrict computeQueue(uint32_t const index) const { return(computeQueue_[index]); }
  vk::Queue const& __restrict transferQueue(uint32_t const index) const { return(transferQueue_[index]); }

  // return image views //
  vk::ImageView const colorImageView() const { return(colorImage_.imageView()); }
  vk::ImageView const guiImageView(uint32_t const index) const { return(guiImage_.resolved[index].imageView()); }
  vk::ImageView const lastColorImageView() const { return(lastColorImage_.imageView()); }

  vk::ImageView const colorVolumetricDownResCheckeredImageView() const { return(colorVolumetricImage_.checkered.imageView()); }
  vk::ImageView const colorVolumetricDownResImageView() const { return(colorVolumetricImage_.resolved.imageView()); }
  vk::ImageView const colorVolumetricImageView() const { return(colorVolumetricImage_.upsampled.imageView()); }

  vk::ImageView const colorReflectionDownResCheckeredImageView() const { return(colorReflectionImage_.checkered.imageView()); }
  vk::ImageView const colorReflectionDownResImageView() const { return(colorReflectionImage_.resolved.imageView()); }
  vk::ImageView const colorReflectionImageView() const { return(colorReflectionImage_.upsampled.imageView()); }
  
  vk::ImageView const offscreenImageView() const { return(offscreenImage_.resolved.imageView()); }

  vk::ImageView const depthImageView() const { return(depthImage_.imageView()); }
  vk::ImageView const depthResolvedImageView(uint32_t const index) const { return(depthImageResolve_[index].imageView()); }
  
  // return images //
  vku::ColorAttachmentImage&	colorImage() { return(colorImage_); }
  vku::ColorAttachmentImage&	mouseImage() { return(mouseImage_.resolved); }
  vku::ColorAttachmentImage&	lastColorImage() { return(lastColorImage_); }
  vku::TextureImageStorage2D&	colorVolumetricDownResCheckeredImage() { return(colorVolumetricImage_.checkered); }
  vku::TextureImageStorage2D&	colorReflectionDownResCheckeredImage() { return(colorReflectionImage_.checkered); }
  vku::ColorAttachmentImage&	colorVolumetricImage() { return(colorVolumetricImage_.upsampled); }
  vku::ColorAttachmentImage&	colorReflectionImage() { return(colorReflectionImage_.upsampled); }
  vku::ColorAttachmentImage&	offscreenImage() { return(offscreenImage_.resolved); }
  vku::ColorAttachmentImage&	guiImage(uint32_t const index) { return(guiImage_.resolved[index]); }
  vku::DepthImage&				depthResolvedImage(uint32_t const index) { return(depthImageResolve_[index]); }

  /// Return the format of the back buffer.
  vk::Format depthImageFormat() const { return depthImage_.format(); }

  /// Return the format of the back buffer.
  vk::Format swapchainImageFormat() const { return swapchainImageFormat_; }

  /// Return the colour space of the back buffer (Usually sRGB)
  vk::ColorSpaceKHR swapchainColorSpace() const { return swapchainColorSpace_; }

  /// Return the swapchain object
  vk::SwapchainKHR const& __restrict swapchain() const { return(*swapchain_); }

  /// Return the views of the swap chain images
  std::vector<vk::ImageView> const& __restrict imageViews() const { return(imageViews_); }

  /// Return the swap chain images
  std::vector<vk::Image> const& __restrict images() const { return(images_); }

  /// Return a defult command Pool to use to create new command buffers.
  vk::CommandPool const& __restrict commandPool(eCommandPools const index) const { return(*commandPool_[index]); }

  /// Return the number of swap chain images.
  int numImageIndices() const { return (int)images_.size(); }

  bool const isFullScreenExclusive() const { return(bFullScreenExclusiveOn); }
  bool const isHDR() const { return(bHDROn); }

private:
	vk::Queue 
		transferQueue_[2],
		computeQueue_[2],
		graphicsQueue_;

	static inline int64_t present_task_id{};

	static inline vk::Result _presentResult{};

  vk::Instance instance_;
  vk::SurfaceKHR surface_;
  vk::UniqueSwapchainKHR swapchain_;
  vk::UniqueRenderPass zPass_, downPass_, upPass_, midPass_, overlayPass_, finalPass_, offscreenPass_;
  
  struct semaphores {
	  vk::UniqueSemaphore imageAcquireSemaphore_;
	  vk::UniqueSemaphore commandCompleteSemaphore_;
	  vk::UniqueSemaphore transferCompleteSemaphore_[2];
	  vk::UniqueSemaphore computeCompleteSemaphore_[2];
  } semaphores[2];
  
  vk::UniqueCommandPool		commandPool_[eCommandPools::_size()];

  static constexpr uint32_t const MAX_IMAGE_VIEWS = 2U; // must be equal to 2, double buffered for some odd/even tricks with framebuffers and associated renderpasses

  std::vector<vk::ImageView> imageViews_;
  std::vector<vk::Image> images_;

  vk::UniqueFramebuffer* framebuffers_[eFrameBuffers::_size()] = { nullptr };
  CommandBufferContainer<eComputeBuffers::_size()> computeDrawBuffers_;	// one for transfer, one for computing light, one for computing textures ....
  CommandBufferContainer<1> staticDrawBuffers_;
  CommandBufferContainer<1> dynamicDrawBuffers_;
  CommandBufferContainer<eOverlayBuffers::_size()> overlayDrawBuffers_;	// one for transfer, one for rendering
   CommandBufferContainer<1> presentDrawBuffers_;

  vku::ColorAttachmentImage colorImage_;	  // multisampled only
  vku::ColorAttachmentImage lastColorImage_;  // not antialiased and does not contain GUI, for that use PostAA lastColorImage - cPostProcess.h

  struct {
	  vku::ColorAttachmentImage		multisampled[2];
	  vku::ColorAttachmentImage		resolved[2];
  } guiImage_;

  struct {
	  vku::ColorAttachmentImage		multisampled;
	  vku::ColorAttachmentImage		resolved;
  } offscreenImage_;

  struct {
	  vku::ColorAttachmentImage		multisampled;			
	  vku::ColorAttachmentImage		resolved;			
  } mouseImage_;

  struct {
	  vku::TextureImageStorage2D	checkered;			// half-resolution
	  vku::ColorAttachmentImage		resolved;			// half-resolution
	  vku::ColorAttachmentImage		upsampled;			// full-resolution
  } colorVolumetricImage_;

  struct {
	  vku::TextureImageStorage2D	checkered;			// half-resolution
	  vku::ColorAttachmentImage		resolved;			// half-resolution
	  vku::ColorAttachmentImage		upsampled;			// full-resolution
  } colorReflectionImage_;

  vku::DepthAttachmentImage			depthImage_;
  vku::DepthImage					depthImageResolve_[2];
  vku::StencilAttachmentImage		stencilCheckerboard_[2];

  uint32_t graphicsQueueFamilyIndex_ = 0;
  uint32_t width_;
  uint32_t height_;
  vk::Format swapchainImageFormat_ = vk::Format::eB8G8R8A8Unorm;
  vk::ColorSpaceKHR swapchainColorSpace_ = vk::ColorSpaceKHR::eSrgbNonlinear;
  vk::Device device_;
  // neccessary for swapchain hot recreation:
  vk::PhysicalDevice physicalDevice_;
  HMONITOR monitor_ = nullptr;
  bool vsyncDisabled_ = false;
  bool ok_ = false;

  std::vector<bool> staticCommandsDirty_;
  bool computeCommandsDirty_[2] = { false };
 
  // extensions enabled & active ? //
  bool bFullScreenExclusiveOn = false;
  bool bHDROn = false;
};

} // namespace vku

#endif // VKU_FRAMEWORK_HPP
