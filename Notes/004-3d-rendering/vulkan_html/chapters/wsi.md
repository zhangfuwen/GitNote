<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#wsi">Window System Integration (WSI)</a>
<ul class="sectlevel1">
<li><a href="#_surface">1. Surface</a></li>
<li><a href="#_swapchain">2. Swapchain</a></li>
<li><a href="#_pre_rotation">3. Pre-Rotation</a></li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink: /Notes/004-3d-rendering/vulkan/chapters/wsi.html
---</p>
</div>
<h1 id="wsi" class="sect0">Window System Integration (WSI)</h1>
<div class="paragraph">
<p>Since the Vulkan API can be used without displaying results, WSI is provided through the use of <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#wsi">optional Vulkan extensions</a>. Most implementations will include WSI support. The WSI design was created to abstract each platform&#8217;s windowing mechanism from the core Vulkan API.</p>
</div>
<div class="imageblock">
<div class="content">
<img src="images/wsi_setup.png" alt="wsi_setup">
</div>
</div>
<div class="sect1">
<h2 id="_surface">1. Surface</h2>
<div class="sectionbody">
<div class="paragraph">
<p>The <code>VkSurfaceKHR</code> object is platform agnostic and designed so the rest of the Vulkan API can use it for all WSI operations. It is enabled using the <code>VK_KHR_surface</code> extension.</p>
</div>
<div class="paragraph">
<p>Each platform that supports a Vulkan Surface has its own way to create a <code>VkSurfaceKHR</code> object from its respective platform-specific API.</p>
</div>
<div class="ulist">
<ul>
<li>
<p>Android - <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#vkCreateAndroidSurfaceKHR">vkCreateAndroidSurfaceKHR</a></p>
</li>
<li>
<p>DirectFB - <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#vkCreateDirectFBSurfaceEXT">vkCreateDirectFBSurfaceEXT</a></p>
</li>
<li>
<p>Fuchsia - <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#vkCreateImagePipeSurfaceFUCHSIA">vkCreateImagePipeSurfaceFUCHSIA</a></p>
</li>
<li>
<p>Google Games - <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#vkCreateStreamDescriptorSurfaceGGP">vkCreateStreamDescriptorSurfaceGGP</a></p>
</li>
<li>
<p>iOS - <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#vkCreateIOSSurfaceMVK">vkCreateIOSSurfaceMVK</a></p>
</li>
<li>
<p>macOS - <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#vkCreateMacOSSurfaceMVK">vkCreateMacOSSurfaceMVK</a></p>
</li>
<li>
<p>Metal - <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#vkCreateMetalSurfaceEXT">vkCreateMetalSurfaceEXT</a></p>
</li>
<li>
<p>VI - <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#vkCreateViSurfaceNN">vkCreateViSurfaceNN</a></p>
</li>
<li>
<p>Wayland - <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#vkWaylandSurfaceCreateInfoKHR">vkWaylandSurfaceCreateInfoKHR</a></p>
</li>
<li>
<p>QNX - <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkCreateScreenSurfaceQNX.html">vkCreateScreenSurfaceQNX</a></p>
</li>
<li>
<p>Windows - <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#vkCreateWin32SurfaceKHR">vkCreateWin32SurfaceKHR</a></p>
</li>
<li>
<p>XCB - <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#vkCreateXcbSurfaceKHR">vkCreateXcbSurfaceKHR</a></p>
</li>
<li>
<p>Xlib - <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#vkCreateXlibSurfaceKHR">vkCreateXlibSurfaceKHR</a></p>
</li>
<li>
<p>Direct-to-Display - <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#vkCreateDisplayPlaneSurfaceKHR">vkCreateDisplayPlaneSurfaceKHR</a></p>
</li>
</ul>
</div>
<div class="paragraph">
<p>Once a <code>VkSurfaceKHR</code> is created there are various <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#vkGetPhysicalDeviceSurfaceCapabilitiesKHR">capabilities</a>, <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#vkGetPhysicalDeviceSurfaceFormatsKHR">formats</a>, and <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#vkGetPhysicalDeviceSurfacePresentModesKHR">presentation modes</a> to query for.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_swapchain">2. Swapchain</h2>
<div class="sectionbody">
<div class="paragraph">
<p>The <code>VkSwapchainKHR</code> object provides the ability to present rendering results to a surface through an array of <code>VkImage</code> objects. The swapchain&#8217;s various <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VkPresentModeKHR">present modes</a> determine how the presentation engine is implemented.</p>
</div>
<div class="imageblock">
<div class="content">
<img src="images/wsi_engine.png" alt="wsi_engine">
</div>
</div>
<div class="paragraph">
<p>Khronos' <a href="https://github.com/KhronosGroup/Vulkan-Samples/tree/master/samples/performance/swapchain_images">sample and tutorial</a> explain different considerations to make when creating a swapchain and selecting a presentation mode.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_pre_rotation">3. Pre-Rotation</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Mobile devices can be rotated, therefore the logical orientation of the application window and the physical orientation of the display may not match. Applications need to be able to operate in two modes: <code>portrait</code> and <code>landscape</code>. The difference between these two modes can be simplified to just a change in resolution. However, some display subsystems always work on the &#8220;native&#8221; (or &#8220;physical&#8221;) orientation of the display panel. Since the device has been rotated, to achieve the desired effect the application output must also rotate.</p>
</div>
<div class="paragraph">
<p>In order for your application to get the most out of Vulkan on mobile platforms, such as Android, implementing pre-rotation is a must. There is a <a href="https://android-developers.googleblog.com/2020/02/handling-device-orientation-efficiently.html?m=1">detailed blog post from Google</a> that goes over how to handle the surface rotation by specifying the orientation during swapchain creation and also comes with a <a href="https://github.com/google/vulkan-pre-rotation-demo">standalone example</a>. The <a href="https://github.com/KhronosGroup/Vulkan-Samples">Vulkan-Samples</a> also has both a <a href="https://github.com/KhronosGroup/Vulkan-Samples/tree/master/samples/performance/surface_rotation">great write up</a> of why pre-rotation is a problem as well as <a href="https://github.com/KhronosGroup/Vulkan-Samples/tree/master/samples/performance/surface_rotation">a sample to run</a> that shows a way to solve it in the shader. If using an Adreno GPU powered device, Qualcomm suggests making use of the <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_QCOM_render_pass_transform.html">VK_QCOM_render_pass_transform</a> extension to implement pre-rotation.</p>
</div>
</div>
</div>