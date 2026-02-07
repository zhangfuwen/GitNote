<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#VK_KHR_sampler_ycbcr_conversion">VK_KHR_sampler_ycbcr_conversion</a>
<ul class="sectlevel1">
<li><a href="#multi-planar-formats">1. Multi-planar Formats</a></li>
<li><a href="#_disjoint">2. Disjoint</a></li>
<li><a href="#_copying_memory_to_each_plane">3. Copying memory to each plane</a></li>
<li><a href="#_vksamplerycbcrconversion">4. VkSamplerYcbcrConversion</a></li>
<li><a href="#_combinedimagesamplerdescriptorcount">5. combinedImageSamplerDescriptorCount</a></li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink:/Notes/004-3d-rendering/vulkan/chapters/extensions/VK_KHR_sampler_ycbcr_conversion.html
layout: default
---</p>
</div>
<h1 id="VK_KHR_sampler_ycbcr_conversion" class="sect0">VK_KHR_sampler_ycbcr_conversion</h1>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Promoted to core in Vulkan 1.1</p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>All the examples below use a <code>4:2:0</code> multi-planar Y′C<sub>B</sub>C<sub>R</sub> format for illustration purposes.</p>
</div>
<div class="sect1">
<h2 id="multi-planar-formats">1. Multi-planar Formats</h2>
<div class="sectionbody">
<div class="paragraph">
<p>To represent a Y′C<sub>B</sub>C<sub>R</sub> image for which the Y' (luma) data is stored in plane 0, the C<sub>B</sub> blue chroma difference value ("U") data is stored in plane 1, and the C<sub>R</sub> red chroma difference value ("V") data is stored in plane 2, an application would use the <code>VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM format</code>.</p>
</div>
<div class="paragraph">
<p>The Vulkan specification separately describes each multi-planar format representation and its mapping to each color component. Because the mapping and color conversion is separated from the format, Vulkan uses &#8220;RGB&#8221; color channel notations in the formats, and the conversion then describes the mapping from these channels to the input to the color conversion.</p>
</div>
<div class="paragraph">
<p>This allows, for example, <code>VK_FORMAT_B8G8R8_UNORM</code> images to represent Y′C<sub>B</sub>C<sub>R</sub> texels.</p>
</div>
<div class="ulist">
<ul>
<li>
<p><code>G</code> == <code>Y</code></p>
</li>
<li>
<p><code>B</code> == <code>Cb</code></p>
</li>
<li>
<p><code>R</code> == <code>Cr</code></p>
</li>
</ul>
</div>
<div class="paragraph">
<p>This may require some extra focus when mapping the swizzle components between <code>RGBA</code> and the Y′C<sub>B</sub>C<sub>R</sub> format.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_disjoint">2. Disjoint</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Normally when an application creates a <code>VkImage</code> it only binds it to a single <code>VkDeviceMemory</code> object. If the implementation supports <code>VK_FORMAT_FEATURE_DISJOINT_BIT</code> for a given format then an application can bind multiple disjoint <code>VkDeviceMemory</code> to a single <code>VkImage</code> where each <code>VkDeviceMemory</code> represents a single plane.</p>
</div>
<div class="paragraph">
<p>Image processing operations on Y′C<sub>B</sub>C<sub>R</sub> images often treat channels separately. For example, applying a sharpening operation to the luma channel or selectively denoising luma. Separating the planes allows them to be processed separately or to reuse unchanged plane data for different final images.</p>
</div>
<div class="paragraph">
<p>Using disjoint images follows the same pattern as the normal binding of memory to an image with the use of a few new functions. Here is some pseudo code to represent the new workflow:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">VkImagePlaneMemoryRequirementsInfo imagePlaneMemoryRequirementsInfo = {};
imagePlaneMemoryRequirementsInfo.planeAspect = VK_IMAGE_ASPECT_PLANE_0_BIT;

VkImageMemoryRequirementsInfo2 imageMemoryRequirementsInfo2 = {};
imageMemoryRequirementsInfo2.pNext = &amp;imagePlaneMemoryRequirementsInfo;
imageMemoryRequirementsInfo2.image = myImage;

// Get memory requirement for each plane
VkMemoryRequirements2 memoryRequirements2 = {};
vkGetImageMemoryRequirements2(device, &amp;imageMemoryRequirementsInfo2, &amp;memoryRequirements2);

// Allocate plane 0 memory
VkMemoryAllocateInfo memoryAllocateInfo = {};
memoryAllocateInfo.allocationSize       = memoryRequirements2.memoryRequirements.size;
vkAllocateMemory(device, &amp;memoryAllocateInfo, nullptr, &amp;disjointMemoryPlane0));

// Allocate the same for each plane

// Bind plane 0 memory
VkBindImagePlaneMemoryInfo bindImagePlaneMemoryInfo = {};
bindImagePlaneMemoryInfo0.planeAspect               = VK_IMAGE_ASPECT_PLANE_0_BIT;

VkBindImageMemoryInfo bindImageMemoryInfo = {};
bindImageMemoryInfo.pNext        = &amp;bindImagePlaneMemoryInfo0;
bindImageMemoryInfo.image        = myImage;
bindImageMemoryInfo.memory       = disjointMemoryPlane0;

// Bind the same for each plane

vkBindImageMemory2(device, bindImageMemoryInfoSize, bindImageMemoryInfoArray));</code></pre>
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_copying_memory_to_each_plane">3. Copying memory to each plane</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Even if an application is not using disjoint memory, it still needs to use the <code>VK_IMAGE_ASPECT_PLANE_0_BIT</code> when copying over data to each plane.</p>
</div>
<div class="paragraph">
<p>For example, if an application plans to do a <code>vkCmdCopyBufferToImage</code> to copy over a single <code>VkBuffer</code> to a single non-disjoint <code>VkImage</code> the data, the logic for a <code>YUV420p</code> layout will look partially like:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">VkBufferImageCopy bufferCopyRegions[3];
bufferCopyRegions[0].imageSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
bufferCopyRegions[0].imageOffset                 = {0, 0, 0};
bufferCopyRegions[0].imageExtent.width           = myImage.width;
bufferCopyRegions[0].imageExtent.height          = myImage.height;
bufferCopyRegions[0].imageExtent.depth           = 1;

/// ...

// the Cb component is half the height and width
bufferCopyRegions[1].imageOffset                  = {0, 0, 0};
bufferCopyRegions[1].imageExtent.width            = myImage.width / 2;
bufferCopyRegions[1].imageExtent.height           = myImage.height / 2;
bufferCopyRegions[1].imageSubresource.aspectMask  = VK_IMAGE_ASPECT_PLANE_1_BIT;

/// ...

// the Cr component is half the height and width
bufferCopyRegions[2].imageOffset                  = {0, 0, 0};
bufferCopyRegions[2].imageExtent.width            = myImage.width / 2;
bufferCopyRegions[2].imageExtent.height           = myImage.height / 2;
bufferCopyRegions[2].imageSubresource.aspectMask  = VK_IMAGE_ASPECT_PLANE_2_BIT;

vkCmdCopyBufferToImage(...)</code></pre>
</div>
</div>
<div class="paragraph">
<p>It is worth noting here is that the <code>imageOffset</code> is zero because its base is the plane, not the entire sname:VkImage. So when using the <code>imageOffset</code> make sure to start from base of the plane and not always plane 0.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_vksamplerycbcrconversion">4. VkSamplerYcbcrConversion</h2>
<div class="sectionbody">
<div class="paragraph">
<p>The <code>VkSamplerYcbcrConversion</code> describes all the &#8220;out of scope explaining here&#8221; aspects of Y′C<sub>B</sub>C<sub>R</sub> conversion which are described in the <a href="https://www.khronos.org/registry/DataFormat/specs/1.3/dataformat.1.3.html#_introduction_to_color_conversions">Khronos Data Format Specification</a>. The values set here are dependent on the input Y′C<sub>B</sub>C<sub>R</sub> data being obtained and how to do the conversion to RGB color spacce.</p>
</div>
<div class="paragraph">
<p>Here is some pseudo code to help give an idea of how to use it from the API point of view:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">// Create conversion object that describes how to have the implementation do the {YCbCr} conversion
VkSamplerYcbcrConversion samplerYcbcrConversion;
VkSamplerYcbcrConversionCreateInfo samplerYcbcrConversionCreateInfo = {};
// ...
vkCreateSamplerYcbcrConversion(device, &amp;samplerYcbcrConversionCreateInfo, nullptr, &amp;samplerYcbcrConversion));

VkSamplerYcbcrConversionInfo samplerYcbcrConversionInfo = {};
samplerYcbcrConversionInfo.conversion = samplerYcbcrConversion;

// Create an ImageView with conversion
VkImageViewCreateInfo imageViewInfo = {};
imageViewInfo.pNext = &amp;samplerYcbcrConversionInfo;
// ...
vkCreateImageView(device, &amp;imageViewInfo, nullptr, &amp;myImageView));

// Create a sampler with conversion
VkSamplerCreateInfo samplerInfo = {};
samplerInfo.pNext = &amp;samplerYcbcrConversionInfo;
// ...
vkCreateSampler(device, &amp;samplerInfo, nullptr, &amp;mySampler));</code></pre>
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_combinedimagesamplerdescriptorcount">5. combinedImageSamplerDescriptorCount</h2>
<div class="sectionbody">
<div class="paragraph">
<p>An important value to monitor is the <code>combinedImageSamplerDescriptorCount</code> which describes how many descriptor an implementation uses for each multi-planar format. This means for <code>VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM</code> an implementation can use 1, 2, or 3 descriptors for each combined image sampler used.</p>
</div>
<div class="paragraph">
<p>All descriptors in a binding use the same maximum <code>combinedImageSamplerDescriptorCount</code> descriptors to allow implementations to use a uniform stride for dynamic indexing of the descriptors in the binding.</p>
</div>
<div class="paragraph">
<p>For example, consider a descriptor set layout binding with two descriptors and immutable samplers for multi-planar formats that have <code>VkSamplerYcbcrConversionImageFormatProperties::combinedImageSamplerDescriptorCount</code> values of <code>2</code> and <code>3</code> respectively. There are two descriptors in the binding and the maximum <code>combinedImageSamplerDescriptorCount</code> is <code>3</code>, so descriptor sets with this layout consume <code>6</code> descriptors from the descriptor pool. To create a descriptor pool that allows allocating <code>4</code> descriptor sets with this layout, <code>descriptorCount</code> must be at least <code>24</code>.</p>
</div>
<div class="paragraph">
<p>Some pseudo code how to query for the <code>combinedImageSamplerDescriptorCount</code>:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">VkSamplerYcbcrConversionImageFormatProperties samplerYcbcrConversionImageFormatProperties = {};

VkImageFormatProperties imageFormatProperties   = {};
VkImageFormatProperties2 imageFormatProperties2 = {};
// ...
imageFormatProperties2.pNext                 = &amp;samplerYcbcrConversionImageFormatProperties;
imageFormatProperties2.imageFormatProperties = imageFormatProperties;

VkPhysicalDeviceImageFormatInfo2 imageFormatInfo = {};
// ...
imageFormatInfo.format = formatToQuery;
vkGetPhysicalDeviceImageFormatProperties2(physicalDevice, &amp;imageFormatInfo, &amp;imageFormatProperties2));

printf("combinedImageSamplerDescriptorCount = %u\n", samplerYcbcrConversionImageFormatProperties.combinedImageSamplerDescriptorCount);</code></pre>
</div>
</div>
</div>
</div>