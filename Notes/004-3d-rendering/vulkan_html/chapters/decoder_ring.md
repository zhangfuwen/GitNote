<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#decoder-ring">Vulkan Decoder Ring</a></li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink: /Notes/004-3d-rendering/vulkan/chapters/decoder_ring.html
---</p>
</div>
<h1 id="decoder-ring" class="sect0">Vulkan Decoder Ring</h1>
<div class="paragraph">
<p>This section provides a mapping between the Vulkan term for a concept and the terminology used in other APIs. It is organized in alphabetical order by Vulkan term. If you are searching for the Vulkan equivalent of a concept used in an API you know, you can find the term you know in this list and then search the <a href="vulkan_spec.html#vulkan-spec">Vulkan specification</a> for the corresponding Vulkan term.</p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Not everything will be a perfect 1:1 match, the goal is to give a rough idea where to start looking in the spec.</p>
</div>
</td>
</tr>
</table>
</div>
<table class="tableblock frame-all grid-all stretch">
<colgroup>
<col style="width: 25%;">
<col style="width: 25%;">
<col style="width: 25%;">
<col style="width: 25%;">
</colgroup>
<thead>
<tr>
<th class="tableblock halign-left valign-top"><strong>Vulkan</strong></th>
<th class="tableblock halign-left valign-top"><strong>GL,GLES</strong></th>
<th class="tableblock halign-left valign-top"><strong>DirectX</strong></th>
<th class="tableblock halign-left valign-top"><strong>Metal</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">buffer device address</p></td>
<td class="tableblock halign-left valign-top"></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">GPU virtual address</p></td>
<td class="tableblock halign-left valign-top"></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">buffer view, texel buffer</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">texture buffer</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">typed buffer SRV, typed buffer UAV</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">texture buffer</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">color attachments</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">color attachments</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">render target</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">color attachments or render target</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">command buffer</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">part of context, display list, NV_command_list</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">command list</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">command buffer</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">command pool</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">part of context</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">command allocator</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">command queue</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">conditional rendering</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">conditional rendering</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">predication</p></td>
<td class="tableblock halign-left valign-top"></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">depth/stencil attachment</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">depth Attachment and stencil Attachment</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">depth/stencil view</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">depth attachment and stencil attachment, depth render target and stencil render target</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">descriptor</p></td>
<td class="tableblock halign-left valign-top"></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">descriptor</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">argument</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">descriptor pool</p></td>
<td class="tableblock halign-left valign-top"></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">descriptor heap</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">heap</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">descriptor set</p></td>
<td class="tableblock halign-left valign-top"></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">descriptor table</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">argument buffer</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">descriptor set layout binding, push descriptor</p></td>
<td class="tableblock halign-left valign-top"></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">root parameter</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">argument in shader parameter list</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">device group</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">implicit (E.g. SLI,CrossFire)</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">multi-adapter device</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">peer group</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">device memory</p></td>
<td class="tableblock halign-left valign-top"></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">heap</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">placement heap</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">event</p></td>
<td class="tableblock halign-left valign-top"></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">split barrier</p></td>
<td class="tableblock halign-left valign-top"></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">fence</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">fence, sync</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock"><code>ID3D12Fence::SetEventOnCompletion</code></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">completed handler, <code>-[MTLCommandBuffer waitUntilComplete]</code></p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">fragment shader</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">fragment shader</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">pixel shader</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">fragment shader or fragment function</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">fragment shader interlock</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock"><a href="https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_fragment_shader_interlock.txt">GL_ARB_fragment_shader_interlock</a></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">rasterizer order view (ROV)</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">raster order group</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">framebuffer</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">framebuffer object</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">collection of resources</p></td>
<td class="tableblock halign-left valign-top"></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">heap</p></td>
<td class="tableblock halign-left valign-top"></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">pool</p></td>
<td class="tableblock halign-left valign-top"></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">image</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">texture and renderbuffer</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">texture</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">texture</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">image layout</p></td>
<td class="tableblock halign-left valign-top"></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">resource state</p></td>
<td class="tableblock halign-left valign-top"></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">image tiling</p></td>
<td class="tableblock halign-left valign-top"></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">image layout, swizzle</p></td>
<td class="tableblock halign-left valign-top"></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">image view</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">texture view</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">render target view, depth/stencil view, shader resource view, unordered access view</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">texture view</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">interface matching (<code>in</code>/<code>out</code>)</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">varying (<a href="https://www.khronos.org/registry/OpenGL/specs/gl/GLSLangSpec.4.20.pdf">removed in GLSL 4.20</a>)</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">Matching semantics</p></td>
<td class="tableblock halign-left valign-top"></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">invocation</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">invocation</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">thread, lane</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">thread, lane</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">layer</p></td>
<td class="tableblock halign-left valign-top"></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">slice</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">slice</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">logical device</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">context</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">device</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">device</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">memory type</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">automatically managed, <a href="https://www.khronos.org/registry/OpenGL/extensions/APPLE/APPLE_texture_range.txt">texture storage hint</a>, <a href="https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_buffer_storage.txt">buffer storage</a></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">heap type, CPU page property</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">storage mode, CPU cache mode</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">multiview rendering</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">multiview rendering</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">view instancing</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">vertex amplification</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">physical device</p></td>
<td class="tableblock halign-left valign-top"></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">adapter, node</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">device</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">pipeline</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">state and program or program pipeline</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">pipeline state</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">pipeline state</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">pipeline barrier, memory barrier</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">texture barrier, memory barrier</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">resource barrier</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">texture barrier, memory barrier</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">pipeline layout</p></td>
<td class="tableblock halign-left valign-top"></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">root signature</p></td>
<td class="tableblock halign-left valign-top"></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">queue</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">part of context</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">command queue</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">command queue</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">semaphore</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">fence, sync</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">fence</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">fence, event</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">shader module</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">shader object</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">resulting <code>ID3DBlob</code> from <code>D3DCompileFromFile</code></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">shader library</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">shading rate attachment</p></td>
<td class="tableblock halign-left valign-top"></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">shading rate image</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">rasterization rate map</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">sparse block</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">sparse block</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">tile</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">sparse tile</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">sparse image</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">sparse texture</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">reserved resource (D12), tiled resource (D11)</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">sparse texture</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">storage buffer</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">shader storage buffer</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">raw or structured buffer UAV</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">buffer in <code>device</code> address space</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">subgroup</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">subgroup</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">wave</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">SIMD-group, quadgroup</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">surface</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">HDC, GLXDrawable, EGLSurface</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">window</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">layer</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">swapchain</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">Part of HDC, GLXDrawable, EGLSurface</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">swapchain</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">layer</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">swapchain image</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">default framebuffer</p></td>
<td class="tableblock halign-left valign-top"></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">drawable texture</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">task shader</p></td>
<td class="tableblock halign-left valign-top"></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">amplification shader</p></td>
<td class="tableblock halign-left valign-top"></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">tessellation control shader</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">tessellation control shader</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">hull shader</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">tessellation compute kernel</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">tessellation evaluation shader</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">tessellation evaluation shader</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">domain shader</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">post-tessellation vertex shader</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">timeline semaphore</p></td>
<td class="tableblock halign-left valign-top"></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">D3D12 fence</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">event</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">transform feedback</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">transform feedback</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">stream-out</p></td>
<td class="tableblock halign-left valign-top"></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">uniform buffer</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">uniform buffer</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">constant buffer views (CBV)</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">buffer in <code>constant</code> address space</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">workgroup</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">workgroup</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">threadgroup</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">threadgroup</p></td>
</tr>
</tbody>
</table>