<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#mapping-data-to-shaders">Mapping Data to Shaders</a>
<ul class="sectlevel1">
<li><a href="#input-attributes">1. Input Attributes</a></li>
<li><a href="#descriptors">2. Descriptors</a>
<ul class="sectlevel2">
<li><a href="#_example">2.1. Example</a></li>
<li><a href="#descriptor-types">2.2. Descriptor types</a></li>
</ul>
</li>
<li><a href="#push-constants">3. Push Constants</a></li>
<li><a href="#specialization-constants">4. Specialization Constants</a>
<ul class="sectlevel2">
<li><a href="#_example_2">4.1. Example</a></li>
<li><a href="#_3_types_of_specialization_constants_usages">4.2. 3 Types of Specialization Constants Usages</a></li>
</ul>
</li>
<li><a href="#physical-storage-buffer">5. Physical Storage Buffer</a></li>
<li><a href="#_limits">6. Limits</a></li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink:/Notes/004-3d-rendering/vulkan/chapters/mapping_data_to_shaders.html
layout: default
---</p>
</div>
<h1 id="mapping-data-to-shaders" class="sect0">Mapping Data to Shaders</h1>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>All SPIR-V assembly was generated with glslangValidator</p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>This chapter goes over how to <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#interfaces">interface Vulkan with SPIR-V</a> in order to map data. Using the <code>VkDeviceMemory</code> objects allocated from <code>vkAllocateMemory</code>, it is up to the application to properly map the data from Vulkan such that the SPIR-V shader understands how to consume it correctly.</p>
</div>
<div class="paragraph">
<p>In core Vulkan, there are 5 fundamental ways to map data from your Vulkan application to interface with SPIR-V:</p>
</div>
<div class="ulist">
<ul>
<li>
<p><a href="#input-attributes">Input Attributes</a></p>
</li>
<li>
<p><a href="#descriptors">Descriptors</a></p>
<div class="ulist">
<ul>
<li>
<p><a href="#descriptor-types">Descriptor types</a></p>
<div class="ulist">
<ul>
<li>
<p><a href="#storage-image">VK_DESCRIPTOR_TYPE_STORAGE_IMAGE</a></p>
</li>
<li>
<p><a href="#sampler-and-sampled-image">VK_DESCRIPTOR_TYPE_SAMPLER and VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE</a></p>
</li>
<li>
<p><a href="#combined-image-sampler">VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER</a></p>
</li>
<li>
<p><a href="#uniform-buffer">VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER</a></p>
</li>
<li>
<p><a href="#storage-buffer">VK_DESCRIPTOR_TYPE_STORAGE_BUFFER</a></p>
</li>
<li>
<p><a href="#uniform-texel-buffer">VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER</a></p>
</li>
<li>
<p><a href="#storage-texel-buffer">VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER</a></p>
</li>
<li>
<p><a href="#input-attachment">VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT</a></p>
</li>
</ul>
</div>
</li>
</ul>
</div>
</li>
<li>
<p><a href="#push-constants">Push Constants</a></p>
</li>
<li>
<p><a href="#specialization-constants">Specialization Constants</a></p>
</li>
<li>
<p><a href="#physical-storage-buffer">Physical Storage Buffer</a></p>
</li>
</ul>
</div>
<div class="sect1">
<h2 id="input-attributes">1. Input Attributes</h2>
<div class="sectionbody">
<div class="paragraph">
<p>The only shader stage in core Vulkan that has an input attribute controlled by Vulkan is the vertex shader stage (<code>VK_SHADER_STAGE_VERTEX_BIT</code>). This involves declaring the interface slots when creating the <code>VkPipeline</code> and then binding the <code>VkBuffer</code> before draw time with the data to map. Other shaders stages, such as a fragment shader stage, has input attributes, but the values are determined from the output of the previous stages ran before it.</p>
</div>
<div class="paragraph">
<p>Before calling <code>vkCreateGraphicsPipelines</code> a <code>VkPipelineVertexInputStateCreateInfo</code> struct will need to be filled out with a list of <code>VkVertexInputAttributeDescription</code> mappings to the shader.</p>
</div>
<div class="paragraph">
<p>An example GLSL vertex shader:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">#version 450
layout(location = 0) in vec3 inPosition;

void main() {
    gl_Position = vec4(inPosition, 1.0);
}</code></pre>
</div>
</div>
<div class="paragraph">
<p>There is only a single input attribute at location 0. This can also be seen in the generated SPIR-V assembly:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-swift" data-lang="swift">                Name 18  "inPosition"
                Decorate 18(inPosition) Location 0

            17: TypePointer Input 16(fvec3)
18(inPosition): 17(ptr) Variable Input
            19: 16(fvec3) Load 18(inPosition)</code></pre>
</div>
</div>
<div class="paragraph">
<p>In this example, the following could be used for the <code>VkVertexInputAttributeDescription</code>:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-c" data-lang="c">VkVertexInputAttributeDescription input = {};
input.location = 0;
input.binding  = 0;
input.format   = VK_FORMAT_R32G32B32_SFLOAT; // maps to vec3
input.offset   = 0;</code></pre>
</div>
</div>
<div class="paragraph">
<p>The only thing left to do is bind the vertex buffer and optional index buffer prior to the draw call.</p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Using <code>VK_BUFFER_USAGE_VERTEX_BUFFER_BIT</code> when creating the <code>VkBuffer</code> is what makes it a &#8220;vertex buffer&#8221;</p>
</div>
</td>
</tr>
</table>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-c" data-lang="c">vkBeginCommandBuffer();
// ...
vkCmdBindVertexBuffer();
vkCmdDraw();
// ...
vkCmdBindVertexBuffer();
vkCmdBindIndexBuffer();
vkCmdDrawIndexed();
// ...
vkEndCommandBuffer();</code></pre>
</div>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>More information can be found in the <a href="vertex_input_data_processing.html#vertex-input-data-processing">Vertex Input Data Processing</a> chapter</p>
</div>
</td>
</tr>
</table>
</div>
</div>
</div>
<div class="sect1">
<h2 id="descriptors">2. Descriptors</h2>
<div class="sectionbody">
<div class="paragraph">
<p>A <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#descriptorsets">resource descriptor</a> is the core way to map data such as uniform buffers, storage buffers, samplers, etc. to any shader stage in Vulkan. One way to conceptualize a descriptor is by thinking of it as a pointer to memory that the shader can use.</p>
</div>
<div class="paragraph">
<p>There are various <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VkDescriptorType">descriptor types</a> in Vulkan, each with their own <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#descriptorsets-types">detailed description</a> in what they allow.</p>
</div>
<div class="paragraph">
<p>Descriptors are grouped together in <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#descriptorsets-sets">descriptor sets</a> which get bound to the shader. Even if there is only a single descriptor in the descriptor set, the entire <code>VkDescriptorSet</code> is used when binding to the shader.</p>
</div>
<div class="sect2">
<h3 id="_example">2.1. Example</h3>
<div class="paragraph">
<p>In this example, there are the following 3 descriptor sets:</p>
</div>
<div class="imageblock">
<div class="content">
<img src="images/mapping_data_to_shaders_descriptor_1.png" alt="mapping_data_to_shaders_descriptor_1.png">
</div>
</div>
<div class="paragraph">
<p>The GLSL of the shader:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">// Note - only set 0 and 2 are used in this shader

layout(set = 0, binding = 0) uniform sampler2D myTextureSampler;

layout(set = 0, binding = 2) uniform uniformBuffer0 {
    float someData;
} ubo_0;

layout(set = 0, binding = 3) uniform uniformBuffer1 {
    float moreData;
} ubo_1;

layout(set = 2, binding = 0) buffer storageBuffer {
    float myResults;
} ssbo;</code></pre>
</div>
</div>
<div class="paragraph">
<p>The corresponding SPIR-V assembly:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-swift" data-lang="swift">Decorate 19(myTextureSampler) DescriptorSet 0
Decorate 19(myTextureSampler) Binding 0

MemberDecorate 29(uniformBuffer0) 0 Offset 0
Decorate 29(uniformBuffer0) Block
Decorate 31(ubo_0) DescriptorSet 0
Decorate 31(ubo_0) Binding 2

MemberDecorate 38(uniformBuffer1) 0 Offset 0
Decorate 38(uniformBuffer1) Block
Decorate 40(ubo_1) DescriptorSet 0
Decorate 40(ubo_1) Binding 3

MemberDecorate 44(storageBuffer) 0 Offset 0
Decorate 44(storageBuffer) BufferBlock
Decorate 46(ssbo) DescriptorSet 2
Decorate 46(ssbo) Binding 0</code></pre>
</div>
</div>
<div class="paragraph">
<p>The binding of descriptors is done while recording the command buffer. The descriptors must be bound at the time of a draw/dispatch call. The following is some pseudo code to better represent this:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-c" data-lang="c">vkBeginCommandBuffer();
// ...
vkCmdBindPipeline(); // Binds shader

// One possible way of binding the two sets
vkCmdBindDescriptorSets(firstSet = 0, pDescriptorSets = &amp;descriptor_set_c);
vkCmdBindDescriptorSets(firstSet = 2, pDescriptorSets = &amp;descriptor_set_b);

vkCmdDraw(); // or dispatch
// ...
vkEndCommandBuffer();</code></pre>
</div>
</div>
<div class="paragraph">
<p>The following results would look as followed</p>
</div>
<div class="imageblock">
<div class="content">
<img src="images/mapping_data_to_shaders_descriptor_2.png" alt="mapping_data_to_shaders_descriptor_2.png">
</div>
</div>
</div>
<div class="sect2">
<h3 id="descriptor-types">2.2. Descriptor types</h3>
<div class="paragraph">
<p>The Vulkan Spec has a <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#interfaces-resources-storage-class-correspondence">Shader Resource and Storage Class Correspondence</a> table that describes how each descriptor type needs to be mapped to in SPIR-V.</p>
</div>
<div class="paragraph">
<p>The following shows an example of what GLSL and SPIR-V mapping to each of the <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#descriptorsets-types">descriptor types</a> looks like.</p>
</div>
<div class="paragraph">
<p>For GLSL, more information can be found in the <a href="https://www.khronos.org/registry/OpenGL/specs/gl/GLSLangSpec.4.60.pdf">GLSL Spec - 12.2.4. Vulkan Only: Samplers, Images, Textures, and Buffers</a></p>
</div>
<div class="sect3">
<h4 id="storage-image">2.2.1. Storage Image</h4>
<div class="paragraph">
<p><code>VK_DESCRIPTOR_TYPE_STORAGE_IMAGE</code></p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">// VK_FORMAT_R32_UINT
layout(set = 0, binding = 0, r32ui) uniform uimage2D storageImage;

// example usage for reading and writing in GLSL
const uvec4 texel = imageLoad(storageImage, ivec2(0, 0));
imageStore(storageImage, ivec2(1, 1), texel);</code></pre>
</div>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-swift" data-lang="swift">OpDecorate %storageImage DescriptorSet 0
OpDecorate %storageImage Binding 0

%r32ui        = OpTypeImage %uint 2D 0 0 0 2 R32ui
%ptr          = OpTypePointer UniformConstant %r32ui
%storageImage = OpVariable %ptr UniformConstant</code></pre>
</div>
</div>
</div>
<div class="sect3">
<h4 id="sampler-and-sampled-image">2.2.2. Sampler and Sampled Image</h4>
<div class="paragraph">
<p><code>VK_DESCRIPTOR_TYPE_SAMPLER</code> and <code>VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE</code></p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">layout(set = 0, binding = 0) uniform sampler samplerDescriptor;
layout(set = 0, binding = 1) uniform texture2D sampledImage;

// example usage of using texture() in GLSL
vec4 data = texture(sampler2D(sampledImage,  samplerDescriptor), vec2(0.0, 0.0));</code></pre>
</div>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-swift" data-lang="swift">OpDecorate %sampledImage DescriptorSet 0
OpDecorate %sampledImage Binding 1
OpDecorate %samplerDescriptor DescriptorSet 0
OpDecorate %samplerDescriptor Binding 0

%image        = OpTypeImage %float 2D 0 0 0 1 Unknown
%imagePtr     = OpTypePointer UniformConstant %image
%sampledImage = OpVariable %imagePtr UniformConstant

%sampler           = OpTypeSampler
%samplerPtr        = OpTypePointer UniformConstant %sampler
%samplerDescriptor = OpVariable %samplerPtr UniformConstant

%imageLoad       = OpLoad %image %sampledImage
%samplerLoad     = OpLoad %sampler %samplerDescriptor

%sampleImageType = OpTypeSampledImage %image
%1               = OpSampledImage %sampleImageType %imageLoad %samplerLoad

%textureSampled = OpImageSampleExplicitLod %v4float %1 %coordinate Lod %float_0</code></pre>
</div>
</div>
</div>
<div class="sect3">
<h4 id="combined-image-sampler">2.2.3. Combined Image Sampler</h4>
<div class="paragraph">
<p><code>VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER</code></p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>On some implementations, it <strong>may</strong> be more efficient to sample from an image using a combination of sampler and sampled image that are stored together in the descriptor set in a combined descriptor.</p>
</div>
</td>
</tr>
</table>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">layout(set = 0, binding = 0) uniform sampler2D combinedImageSampler;

// example usage of using texture() in GLSL
vec4 data = texture(combinedImageSampler, vec2(0.0, 0.0));</code></pre>
</div>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-swift" data-lang="swift">OpDecorate %combinedImageSampler DescriptorSet 0
OpDecorate %combinedImageSampler Binding 0

%imageType            = OpTypeImage %float 2D 0 0 0 1 Unknown
%sampleImageType      = OpTypeSampledImage imageType
%ptr                  = OpTypePointer UniformConstant %sampleImageType
%combinedImageSampler = OpVariable %ptr UniformConstant

%load           = OpLoad %sampleImageType %combinedImageSampler
%textureSampled = OpImageSampleExplicitLod %v4float %load %coordinate Lod %float_0</code></pre>
</div>
</div>
</div>
<div class="sect3">
<h4 id="uniform-buffer">2.2.4. Uniform Buffer</h4>
<div class="paragraph">
<p><code>VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER</code></p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Uniform buffers can also have <a href="descriptor_dynamic_offset.html">dynamic offsets at bind time</a> (VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC)</p>
</div>
</td>
</tr>
</table>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">layout(set = 0, binding = 0) uniform uniformBuffer {
    float a;
    int b;
} ubo;

// example of reading from UBO in GLSL
int x = ubo.b + 1;
vec3 y = vec3(ubo.a);</code></pre>
</div>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-swift" data-lang="swift">OpMemberDecorate %uniformBuffer 0 Offset 0
OpMemberDecorate %uniformBuffer 1 Offset 4
OpDecorate %uniformBuffer Block
OpDecorate %ubo DescriptorSet 0
OpDecorate %ubo Binding 0

%uniformBuffer = OpTypeStruct %float %int
%ptr           = OpTypePointer Uniform %uniformBuffer
%ubo           = OpVariable %ptr Uniform</code></pre>
</div>
</div>
</div>
<div class="sect3">
<h4 id="storage-buffer">2.2.5. Storage Buffer</h4>
<div class="paragraph">
<p><code>VK_DESCRIPTOR_TYPE_STORAGE_BUFFER</code></p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Storage buffers can also have <a href="descriptor_dynamic_offset.html">dynamic offsets at bind time</a> (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)</p>
</div>
</td>
</tr>
</table>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">layout(set = 0, binding = 0) buffer storageBuffer {
    float a;
    int b;
} ssbo;

// example of reading and writing SSBO in GLSL
ssbo.a = ssbo.a + 1.0;
ssbo.b = ssbo.b + 1;</code></pre>
</div>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="title">Important</div>
<div class="paragraph">
<p><code>BufferBlock</code> and <code>Uniform</code> would have been seen prior to <a href="extensions/shader_features.html#VK_KHR_storage_buffer_storage_class">VK_KHR_storage_buffer_storage_class</a></p>
</div>
</td>
</tr>
</table>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-swift" data-lang="swift">OpMemberDecorate %storageBuffer 0 Offset 0
OpMemberDecorate %storageBuffer 1 Offset 4
OpDecorate %storageBuffer Block
OpDecorate %ssbo DescriptorSet 0
OpDecorate %ssbo Binding 0

%storageBuffer = OpTypeStruct %float %int
%ptr           = OpTypePointer StorageBuffer %storageBuffer
%ssbo          = OpVariable %ptr StorageBuffer</code></pre>
</div>
</div>
</div>
<div class="sect3">
<h4 id="uniform-texel-buffer">2.2.6. Uniform Texel Buffer</h4>
<div class="paragraph">
<p><code>VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER</code></p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">layout(set = 0, binding = 0) uniform textureBuffer uniformTexelBuffer;

// example of reading texel buffer in GLSL
vec4 data = texelFetch(uniformTexelBuffer, 0);</code></pre>
</div>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-swift" data-lang="swift">OpDecorate %uniformTexelBuffer DescriptorSet 0
OpDecorate %uniformTexelBuffer Binding 0

%texelBuffer        = OpTypeImage %float Buffer 0 0 0 1 Unknown
%ptr                = OpTypePointer UniformConstant %texelBuffer
%uniformTexelBuffer = OpVariable %ptr UniformConstant</code></pre>
</div>
</div>
</div>
<div class="sect3">
<h4 id="storage-texel-buffer">2.2.7. Storage Texel Buffer</h4>
<div class="paragraph">
<p><code>VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER</code></p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">// VK_FORMAT_R8G8B8A8_UINT
layout(set = 0, binding = 0, rgba8ui) uniform uimageBuffer storageTexelBuffer;

// example of reading and writing texel buffer in GLSL
int offset = int(gl_GlobalInvocationID.x);
vec4 data = imageLoad(storageTexelBuffer, offset);
imageStore(storageTexelBuffer, offset, uvec4(0));</code></pre>
</div>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-swift" data-lang="swift">OpDecorate %storageTexelBuffer DescriptorSet 0
OpDecorate %storageTexelBuffer Binding 0

%rgba8ui            = OpTypeImage %uint Buffer 0 0 0 2 Rgba8ui
%ptr                = OpTypePointer UniformConstant %rgba8ui
%storageTexelBuffer = OpVariable %ptr UniformConstant</code></pre>
</div>
</div>
</div>
<div class="sect3">
<h4 id="input-attachment">2.2.8. Input Attachment</h4>
<div class="paragraph">
<p><code>VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT</code></p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">layout (input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput inputAttachment;

// example loading the attachment data in GLSL
vec4 data = subpassLoad(inputAttachment);</code></pre>
</div>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-swift" data-lang="swift">OpDecorate %inputAttachment DescriptorSet 0
OpDecorate %inputAttachment Binding 0
OpDecorate %inputAttachment InputAttachmentIndex 0

%subpass         = OpTypeImage %float SubpassData 0 0 0 2 Unknown
%ptr             = OpTypePointer UniformConstant %subpass
%inputAttachment = OpVariable %ptr UniformConstant</code></pre>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="push-constants">3. Push Constants</h2>
<div class="sectionbody">
<div class="paragraph">
<p>A push constant is a small bank of values accessible in shaders. Push constants allow the application to set values used in shaders without creating buffers or modifying and binding descriptor sets for each update.</p>
</div>
<div class="paragraph">
<p>These are designed for small amount (a few dwords) of high frequency data to be updated per-recording of the command buffer.</p>
</div>
<div class="paragraph">
<p>From a shader perspective, it is similar to a uniform buffer.</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">#version 450

layout(push_constant) uniform myPushConstants {
    vec4 myData;
} myData;</code></pre>
</div>
</div>
<div class="paragraph">
<p>Resulting SPIR-V assembly:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-swift" data-lang="swift">MemberDecorate 13(myPushConstants) 0 Offset 0
Decorate 13(myPushConstants) Block</code></pre>
</div>
</div>
<div class="paragraph">
<p>While recording the command buffer the values of the push constants are decided.</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-c" data-lang="c">vkBeginCommandBuffer();
// ...
vkCmdBindPipeline();

float someData[4] = {0.0, 1.0, 2.0, 3.0};
vkCmdPushConstants(sizeof(float) * 4, someData);

vkCmdDraw();
// ...
vkEndCommandBuffer();</code></pre>
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="specialization-constants">4. Specialization Constants</h2>
<div class="sectionbody">
<div class="paragraph">
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#pipelines-specialization-constants">Specialization constants</a> are a mechanism allowing a constant value in SPIR-V to be specified at <code>VkPipeline</code> creation time. This is powerful as it replaces the idea of doing preprocessor macros in the high level shading language (GLSL, HLSL, etc).</p>
</div>
<div class="sect2">
<h3 id="_example_2">4.1. Example</h3>
<div class="paragraph">
<p>If an application wants to create to <code>VkPipeline</code> where the color value is different for each, a naive approach is to have two shaders:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">// shader_a.frag
#version 450
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(0.0);
}</code></pre>
</div>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">// shader_b.frag
#version 450
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(1.0);
}</code></pre>
</div>
</div>
<div class="paragraph">
<p>Using specialization constants, the decision can instead be made when calling <code>vkCreateGraphicsPipelines</code> to compile the shader. This means there only needs to be a single shader.</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">#version 450
layout (constant_id = 0) const float myColor = 1.0;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(myColor);
}</code></pre>
</div>
</div>
<div class="paragraph">
<p>Resulting SPIR-V assembly:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-spswiftirv" data-lang="spswiftirv">                      Decorate 9(outColor) Location 0
                      Decorate 10(myColor) SpecId 0

                      // 0x3f800000 as decimal which is 1.0 for a 32 bit float
10(myColor): 6(float) SpecConstant 1065353216</code></pre>
</div>
</div>
<div class="paragraph">
<p>With specialization constants, the value is still a constant inside the shader, but for example, if another <code>VkPipeline</code> uses the same shader, but wants to set the <code>myColor</code> value to <code>0.5f</code>, it is possible to do so at runtime.</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">struct myData {
    float myColor = 1.0f;
} myData;

VkSpecializationMapEntry mapEntry = {};
mapEntry.constantID = 0; // matches constant_id in GLSL and SpecId in SPIR-V
mapEntry.offset     = 0;
mapEntry.size       = sizeof(float);

VkSpecializationInfo specializationInfo = {};
specializationInfo.mapEntryCount = 1;
specializationInfo.pMapEntries   = &amp;mapEntry;
specializationInfo.dataSize      = sizeof(myData);
specializationInfo.pData         = &amp;myData;

VkGraphicsPipelineCreateInfo pipelineInfo = {};
pipelineInfo.pStages[fragIndex].pSpecializationInfo = &amp;specializationInfo;

// Create first pipeline with myColor as 1.0
vkCreateGraphicsPipelines(&amp;pipelineInfo);

// Create second pipeline with same shader, but sets different value
myData.myColor = 0.5f;
vkCreateGraphicsPipelines(&amp;pipelineInfo);</code></pre>
</div>
</div>
<div class="paragraph">
<p>The second <code>VkPipeline</code> shader disassembled has the new constant value for <code>myColor</code> of <code>0.5f</code>.</p>
</div>
</div>
<div class="sect2">
<h3 id="_3_types_of_specialization_constants_usages">4.2. 3 Types of Specialization Constants Usages</h3>
<div class="paragraph">
<p>The typical use cases for specialization constants can be best grouped into three different usages.</p>
</div>
<div class="ulist">
<ul>
<li>
<p>Toggling features</p>
<div class="ulist">
<ul>
<li>
<p>Support for a feature in Vulkan isn&#8217;t known until runtime. This usage of specialization constants is to prevent writing two separate shaders, but instead embedding a constant runtime decision.</p>
</li>
</ul>
</div>
</li>
<li>
<p>Improving backend optimizations</p>
<div class="ulist">
<ul>
<li>
<p>The &#8220;backend&#8221; here refers the implementation&#8217;s compiler that takes the resulting SPIR-V and lowers it down to some ISA to run on the device.</p>
</li>
<li>
<p>Constant values allow a set of optimizations such as <a href="https://en.wikipedia.org/wiki/Constant_folding">constant folding</a>, <a href="https://en.wikipedia.org/wiki/Dead_code_elimination">dead code elimination</a>, etc. to occur.</p>
</li>
</ul>
</div>
</li>
<li>
<p>Affecting types and memory sizes</p>
<div class="ulist">
<ul>
<li>
<p>It is possible to set the length of an array or a variable type used through a specialization constant.</p>
</li>
<li>
<p>It is important to notice that a compiler will need to allocate registers depending on these types and sizes. This means it is likely that a pipeline cache will fail if the difference is significant in registers allocated.</p>
</li>
</ul>
</div>
</li>
</ul>
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="physical-storage-buffer">5. Physical Storage Buffer</h2>
<div class="sectionbody">
<div class="paragraph">
<p>The <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_KHR_buffer_device_address.html#_description">VK_KHR_buffer_device_address</a> extension promoted to Vulkan 1.2 adds the ability to have &#8220;pointers in the shader&#8221;. Using the <code>PhysicalStorageBuffer</code> storage class in SPIR-V an application can call <code>vkGetBufferDeviceAddress</code> which will return the <code>VkDeviceAddress</code> to the memory.</p>
</div>
<div class="paragraph">
<p>While this is a way to map data to the shader, it is not a way to interface with the shader. For example, if an application wants to use this with a uniform buffer it would have to create a <code>VkBuffer</code> with both <code>VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT</code> and <code>VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT</code>. From here in this example, Vulkan would use a descriptor to interface with the shader, but could then use the physical storage buffer to update the value after.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_limits">6. Limits</h2>
<div class="sectionbody">
<div class="paragraph">
<p>With all the above examples it is important to be aware that there are <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#limits">limits in Vulkan</a> that expose how much data can be bound at a single time.</p>
</div>
<div class="ulist">
<ul>
<li>
<p>Input Attributes</p>
<div class="ulist">
<ul>
<li>
<p><code>maxVertexInputAttributes</code></p>
</li>
<li>
<p><code>maxVertexInputAttributeOffset</code></p>
</li>
</ul>
</div>
</li>
<li>
<p>Descriptors</p>
<div class="ulist">
<ul>
<li>
<p><code>maxBoundDescriptorSets</code></p>
</li>
<li>
<p>Per stage limit</p>
</li>
<li>
<p><code>maxPerStageDescriptorSamplers</code></p>
</li>
<li>
<p><code>maxPerStageDescriptorUniformBuffers</code></p>
</li>
<li>
<p><code>maxPerStageDescriptorStorageBuffers</code></p>
</li>
<li>
<p><code>maxPerStageDescriptorSampledImages</code></p>
</li>
<li>
<p><code>maxPerStageDescriptorStorageImages</code></p>
</li>
<li>
<p><code>maxPerStageDescriptorInputAttachments</code></p>
</li>
<li>
<p>Per type limit</p>
</li>
<li>
<p><code>maxPerStageResources</code></p>
</li>
<li>
<p><code>maxDescriptorSetSamplers</code></p>
</li>
<li>
<p><code>maxDescriptorSetUniformBuffers</code></p>
</li>
<li>
<p><code>maxDescriptorSetUniformBuffersDynamic</code></p>
</li>
<li>
<p><code>maxDescriptorSetStorageBuffers</code></p>
</li>
<li>
<p><code>maxDescriptorSetStorageBuffersDynamic</code></p>
</li>
<li>
<p><code>maxDescriptorSetSampledImages</code></p>
</li>
<li>
<p><code>maxDescriptorSetStorageImages</code></p>
</li>
<li>
<p><code>maxDescriptorSetInputAttachments</code></p>
</li>
<li>
<p><code>VkPhysicalDeviceDescriptorIndexingProperties</code> if using <a href="extensions/VK_EXT_inline_uniform_block.html#VK_EXT_inline_uniform_block">Descriptor Indexing</a></p>
</li>
<li>
<p><code>VkPhysicalDeviceInlineUniformBlockPropertiesEXT</code> if using <a href="extensions/VK_EXT_inline_uniform_block.html#VK_EXT_inline_uniform_block">Inline Uniform Block</a></p>
</li>
</ul>
</div>
</li>
<li>
<p>Push Constants</p>
<div class="ulist">
<ul>
<li>
<p><code>maxPushConstantsSize</code> - guaranteed at least <code>128</code> bytes on all devices</p>
</li>
</ul>
</div>
</li>
</ul>
</div>
</div>
</div>