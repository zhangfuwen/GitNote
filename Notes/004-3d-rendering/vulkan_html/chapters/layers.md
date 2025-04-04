<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#layers">Layers</a>
<ul class="sectlevel1">
<li><a href="#_using_layers">1. Using Layers</a></li>
<li><a href="#_vulkan_configurator_tool">2. Vulkan Configurator Tool</a></li>
<li><a href="#_device_layers_deprecation">3. Device Layers Deprecation</a></li>
<li><a href="#_creating_a_layer">4. Creating a Layer</a></li>
<li><a href="#_platform_variations">5. Platform Variations</a>
<ul class="sectlevel2">
<li><a href="#_android">5.1. Android</a></li>
<li><a href="#_linux">5.2. Linux</a></li>
<li><a href="#_macos">5.3. MacOS</a></li>
<li><a href="#_windows">5.4. Windows</a></li>
</ul>
</li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink: /Notes/004-3d-rendering/vulkan/chapters/layers.html
---</p>
</div>
<h1 id="layers" class="sect0">Layers</h1>
<div class="paragraph">
<p>Layers are optional components that augment the Vulkan system. They can intercept, evaluate, and modify existing Vulkan functions on their way from the application down to the hardware. Layer properties can be queried from an application with <a href="https://www.khronos.org/registry/vulkan/specs/1.3/html/vkspec.html#vkEnumerateInstanceLayerProperties">vkEnumerateInstanceLayerProperties</a>.</p>
</div>
<div class="sect1">
<h2 id="_using_layers">1. Using Layers</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Layers are packaged as shared libraries that get dynamically loaded in by the loader and inserted between it and the application. The two things needed to use layers are the location of the binary files and which layers to enable. The layers to use can be either explicitly enabled by the application or implicitly enabled by telling the loader to use them. More details about implicit and explicit layers can be found in the <a href="https://github.com/KhronosGroup/Vulkan-Loader/blob/master/loader/LoaderAndLayerInterface.md#implicit-vs-explicit-layers">Loader and Layer Interface</a>.</p>
</div>
<div class="paragraph">
<p>The <a href="https://vulkan.lunarg.com/sdk/home">Vulkan SDK</a> contains a <a href="https://vulkan.lunarg.com/doc/sdk/latest/windows/layer_configuration.html">layer configuration document</a> that is very specific to how to discover and configure layers on each of the platforms.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_vulkan_configurator_tool">2. Vulkan Configurator Tool</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Developers on Windows, Linux, and macOS can use the Vulkan Configurator, vkconfig, to enable explicit layers and disable implicit layers as well as change layer settings from a graphical user interface.
Please see the <a href="https://vulkan.lunarg.com/doc/sdk/latest/windows/vkconfig.html">Vulkan Configurator documentation</a> in the Vulkan SDK for more information on using the Vulkan Configurator.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_device_layers_deprecation">3. Device Layers Deprecation</h2>
<div class="sectionbody">
<div class="paragraph">
<p>There used to be both instance layers and device layers, but device layers were <a href="https://www.khronos.org/registry/vulkan/specs/1.3/html/vkspec.html#extendingvulkan-layers-devicelayerdeprecation">deprecated</a> early in Vulkan&#8217;s life and should be avoided.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_creating_a_layer">4. Creating a Layer</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Anyone can create a layer as long as it follows the <a href="https://github.com/KhronosGroup/Vulkan-Loader/blob/master/loader/LoaderAndLayerInterface.md#loader-and-layer-interface">loader to layer interface</a> which is how the loader and layers agree to communicate with each other.</p>
</div>
<div class="paragraph">
<p>LunarG provides a framework for layer creation called the <a href="https://github.com/LunarG/VulkanTools/tree/master/layer_factory">Layer Factory</a> to help develop new layers (<a href="https://www.youtube.com/watch?v=gVT7nyXz6M8&amp;t=5m22s">Video presentation</a>).
The layer factory hides the majority of the loader-layer interface, layer boilerplate, setup and initialization, and complexities of layer development.
During application development, the ability to easily create a layer to aid in debugging your application can be useful.
For more information, see the <a href="https://github.com/LunarG/VulkanTools/blob/master/layer_factory/README.md">Vulkan Layer Factory documentation</a>.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_platform_variations">5. Platform Variations</h2>
<div class="sectionbody">
<div class="paragraph">
<p>The way to load a layer in implicitly varies between loader and platform.</p>
</div>
<div class="sect2">
<h3 id="_android">5.1. Android</h3>
<div class="paragraph">
<p>As of Android P (Android 9 / API level 28), if a device is in a debuggable state such that <code>getprop ro.debuggable</code> <a href="http://androidxref.com/9.0.0_r3/xref/frameworks/native/vulkan/libvulkan/layers_extensions.cpp#454">returns 1</a>, then the loader will look in <a href="http://androidxref.com/9.0.0_r3/xref/frameworks/native/vulkan/libvulkan/layers_extensions.cpp#67">/data/local/debug/vulkan</a>.</p>
</div>
<div class="paragraph">
<p>Starting in Android P (Android 9 / API level 28) implicit layers can be <a href="https://developer.android.com/ndk/guides/graphics/validation-layer#vl-adb">pushed using ADB</a> if the application was built in debug mode.</p>
</div>
<div class="paragraph">
<p>There is no way other than the options above to use implicit layers.</p>
</div>
</div>
<div class="sect2">
<h3 id="_linux">5.2. Linux</h3>
<div class="paragraph">
<p>The <a href="https://vulkan.lunarg.com/doc/sdk/latest/linux/layer_configuration.html">Vulkan SDK</a> explains how to use implicit layers on Linux.</p>
</div>
</div>
<div class="sect2">
<h3 id="_macos">5.3. MacOS</h3>
<div class="paragraph">
<p>The <a href="https://vulkan.lunarg.com/doc/sdk/latest/mac/layer_configuration.html">Vulkan SDK</a> explains how to use implicit layers on MacOS.</p>
</div>
</div>
<div class="sect2">
<h3 id="_windows">5.4. Windows</h3>
<div class="paragraph">
<p>The <a href="https://vulkan.lunarg.com/doc/sdk/latest/windows/layer_configuration.html">Vulkan SDK</a> explains how to use implicit layers on Windows.</p>
</div>
</div>
</div>
</div>