<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#ray-tracing">Ray Tracing</a>
<ul class="sectlevel1">
<li><a href="#VK_KHR_acceleration_structure">1. VK_KHR_acceleration_structure</a></li>
<li><a href="#VK_KHR_ray_tracing_pipeline">2. VK_KHR_ray_tracing_pipeline</a></li>
<li><a href="#VK_KHR_ray_query">3. VK_KHR_ray_query</a></li>
<li><a href="#VK_KHR_pipeline_library">4. VK_KHR_pipeline_library</a></li>
<li><a href="#VK_KHR_deferred_host_operations">5. VK_KHR_deferred_host_operations</a></li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink:/Notes/004-3d-rendering/vulkan/chapters/extensions/ray_tracing.html
layout: default
---</p>
</div>
<h1 id="ray-tracing" class="sect0">Ray Tracing</h1>
<div class="paragraph">
<p>A set of five interrelated extensions provide ray tracing support in the Vulkan API.</p>
</div>
<div class="ulist">
<ul>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_KHR_acceleration_structure.html">VK_KHR_acceleration_structure</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_KHR_ray_tracing_pipeline.html">VK_KHR_ray_tracing_pipeline</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_KHR_ray_query.html">VK_KHR_ray_query</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_KHR_pipeline_library.html">VK_KHR_pipeline_library</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_KHR_deferred_host_operations.html">VK_KHR_deferred_host_operations</a></p>
</li>
</ul>
</div>
<div class="paragraph">
<p>Additional SPIR-V and GLSL extensions also expose the necessary programmable functionality for shaders:</p>
</div>
<div class="ulist">
<ul>
<li>
<p><a href="http://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/KHR/SPV_KHR_ray_tracing.html">SPV_KHR_ray_tracing</a></p>
</li>
<li>
<p><a href="http://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/KHR/SPV_KHR_ray_query.html">SPV_KHR_ray_query</a></p>
</li>
<li>
<p><a href="https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GLSL_EXT_ray_tracing.txt">GLSL_EXT_ray_tracing</a></p>
</li>
<li>
<p><a href="https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GLSL_EXT_ray_query.txt">GLSL_EXT_ray_query</a></p>
</li>
<li>
<p><a href="https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GLSL_EXT_ray_flags_primitive_culling.txt">GLSL_EXT_ray_flags_primitive_culling</a></p>
</li>
</ul>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Many ray tracing applications require large contiguous memory
allocations. Due to the limited size of the address space, this can prove
challenging on 32-bit systems. Whilst implementations are free to expose ray
tracing extensions on 32-bit systems, applications may encounter intermittent
memory-related issues such as allocation failures due to fragmentation.
Additionally, some implementations may opt not to expose ray tracing
extensions on 32-bit drivers.</p>
</div>
</td>
</tr>
</table>
</div>
<div class="sect1">
<h2 id="VK_KHR_acceleration_structure">1. VK_KHR_acceleration_structure</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Acceleration structures are an implementation-dependent opaque representation
of geometric objects, which are used for ray tracing.
By building objects into acceleration structures, ray tracing can be performed
against a known data layout, and in an efficient manner.
The <code>VK_KHR_acceleration_structure</code> extension introduces functionality to build
and copy acceleration structures, along with functionality to support
serialization to/from memory.</p>
</div>
<div class="paragraph">
<p>Acceleration structures are required for both ray pipelines
(<code>VK_KHR_ray_tracing_pipeline</code>) and ray queries (<code>VK_KHR_ray_query</code>).</p>
</div>
<div class="paragraph">
<p>To create an acceleration structure:</p>
</div>
<div class="ulist">
<ul>
<li>
<p>Populate an instance of <code>VkAccelerationStructureBuildGeometryInfoKHR</code> with
the acceleration structure type, geometry types, counts, and maximum sizes.
The geometry data does not need to be populated at this point.</p>
</li>
<li>
<p>Call <code>vkGetAccelerationStructureBuildSizesKHR</code> to get the memory size
requirements to perform a build.</p>
</li>
<li>
<p>Allocate buffers of sufficient size to hold the acceleration structure
(<code>VkAccelerationStructureBuildSizesKHR::accelerationStructureSize</code>) and build
scratch buffer (<code>VkAccelerationStructureBuildSizesKHR::buildScratchSize</code>)</p>
</li>
<li>
<p>Call <code>vkCreateAccelerationStructureKHR</code> to create an acceleration structure
at a specified location within a buffer</p>
</li>
<li>
<p>Call <code>vkCmdBuildAccelerationStructuresKHR</code> to build the acceleration structure.
The previously populated <code>VkAccelerationStructureBuildGeometryInfoKHR</code> should
be used as a parameter here, along with the destination acceleration structure
object, build scratch buffer, and geometry data pointers (for vertices,
indices and transforms)</p>
</li>
</ul>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_KHR_ray_tracing_pipeline">2. VK_KHR_ray_tracing_pipeline</h2>
<div class="sectionbody">
<div class="paragraph">
<p>The <code>VK_KHR_ray_tracing_pipeline</code> extension introduces ray tracing pipelines.
This new form of rendering pipeline is independent of the traditional
rasterization pipeline. Ray tracing pipelines utilize a dedicated set of shader
stages, distinct from the traditional vertex/geometry/fragment stages. Ray tracing
pipelines also utilize dedicated commands to submit rendering work
(<code>vkCmdTraceRaysKHR</code> and <code>vkCmdTraceRaysIndirectKHR</code>). These commands can be
regarded as somewhat analagous to the drawing commands in traditional
rasterization pipelines (<code>vkCmdDraw</code> and <code>vkCmdDrawIndirect</code>).</p>
</div>
<div class="paragraph">
<p>To trace rays:</p>
</div>
<div class="ulist">
<ul>
<li>
<p>Bind a ray tracing pipeline using <code>vkCmdBindPipeline</code> with
<code>VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR</code></p>
</li>
<li>
<p>Call <code>vkCmdTraceRaysKHR</code> or <code>vkCmdTraceRaysIndirectKHR</code></p>
</li>
</ul>
</div>
<div class="paragraph">
<p>Ray tracing pipelines introduce several new shader domains. These are described
below:</p>
</div>
<div class="imageblock">
<div class="content">
<img src="https://www.khronos.org/assets/uploads/blogs/2020-The-ray-tracing-mechanism-achieved-through-the-five-shader-stages-2.jpg" alt="Ray Tracing Shaders">
</div>
</div>
<div class="ulist">
<ul>
<li>
<p>Ray generation shader represents the starting point for ray tracing. The ray tracing commands
(<code>vkCmdTraceRaysKHR</code> and <code>vkCmdTraceRaysIndirectKHR</code>) launch a grid of shader invocations,
similar to compute shaders. A ray generation shader constructs rays and begins tracing via
the invocation of traceRayEXT(). Additionally, it processes the results from the hit group.</p>
</li>
<li>
<p>Closest hit shaders are executed when the ray intersects the closest geometry. An application
can support any number of closest hit shaders. They are typically used for carrying out
lighting calculations and can recursively trace additional rays.</p>
</li>
<li>
<p>Miss shaders are executed instead of a closest hit shader when a ray does not intersect any
geometry during traversal. A common use for a miss shader is to sample an environment map.</p>
</li>
<li>
<p>The built-in intersection test is a ray-triangle test. Intersection shaders allow for custom
intersection handling.</p>
</li>
<li>
<p>Similar to the closest hit shader, any-hit shaders are executed after an intersection is
reported. The difference is that an any-hit shader are be invoked for any intersection in
the ray interval defined by [tmin, tmax] and not the closest one to the origin of the ray.
The any-hit shader is used to filter an intersection and therefore is often used to
implement alpha-testing.</p>
</li>
</ul>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_KHR_ray_query">3. VK_KHR_ray_query</h2>
<div class="sectionbody">
<div class="paragraph">
<p>The <code>VK_KHR_ray_query</code> extension provides support for tracing rays from all
shader types, including graphics, compute, and ray tracing pipelines.</p>
</div>
<div class="paragraph">
<p>Ray query requires that ray traversal code is explicitly included within the
shader. This differs from ray tracing pipelines, where ray generation,
intersection testing and handling of ray-geometry hits are represented as
separate shader stages. Consequently, whilst ray query allows rays to be traced
from a wider range of shader stages, it also restricts the range of optimizations
that a Vulkan implementation might apply to the scheduling and tracing of rays.</p>
</div>
<div class="paragraph">
<p>The extension does not introduce additional API entry-points. It simply provides
API support for the related SPIR-V and GLSL extensions (<code>SPV_KHR_ray_query</code> and
<code>GLSL_EXT_ray_query</code>).</p>
</div>
<div class="paragraph">
<p>The functionality provided by <code>VK_KHR_ray_query</code> is complementary to that
provided by <code>VK_KHR_ray_tracing_pipeline</code>, and the two extensions can be used
together.</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">rayQueryEXT rq;

rayQueryInitializeEXT(rq, accStruct, gl_RayFlagsNoneEXT, 0, origin, tMin, direction, tMax);

while(rayQueryProceedEXT(rq)) {
        if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCandidateIntersectionTriangleEXT) {
                //...
                rayQueryConfirmIntersectionEXT(rq);
        }
}

if (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
        //...
}</code></pre>
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_KHR_pipeline_library">4. VK_KHR_pipeline_library</h2>
<div class="sectionbody">
<div class="paragraph">
<p><code>VK_KHR_pipeline_library</code> introduces pipeline libraries. A pipeline library is
a special pipeline that was created using the <code>VK_PIPELINE_CREATE_LIBRARY_BIT_KHR</code>
and cannot be bound and used directly. Instead, these are pipelines that
represent a collection of shaders, shader groups and related state which can be
linked into other pipelines.</p>
</div>
<div class="paragraph">
<p><code>VK_KHR_pipeline_library</code> does not introduce any new API functions directly, or
define how to create a pipeline library. Instead, this functionality is left to
other extensions which make use of the functionality provided by
<code>VK_KHR_pipeline_library</code>.
Currently, the only example of this is <code>VK_KHR_ray_tracing_pipeline</code>.
<code>VK_KHR_pipeline_library</code> was defined as a separate extension to allow for the
possibility of using the same functionality in other extensions in the future
without introducing a dependency on the ray tracing extensions.</p>
</div>
<div class="paragraph">
<p>To create a ray tracing pipeline library:</p>
</div>
<div class="ulist">
<ul>
<li>
<p>Set <code>VK_PIPELINE_CREATE_LIBRARY_BIT_KHR</code> in <code>VkRayTracingPipelineCreateInfoKHR::flags</code>
when calling <code>vkCreateRayTracingPipelinesKHR</code></p>
</li>
</ul>
</div>
<div class="paragraph">
<p>To link ray tracing pipeline libraries into a full pipeline:</p>
</div>
<div class="ulist">
<ul>
<li>
<p>Set <code>VkRayTracingPipelineCreateInfoKHR::pLibraryInfo</code> to point to an instance
of <code>VkPipelineLibraryCreateInfoKHR</code></p>
</li>
<li>
<p>Populate <code>VkPipelineLibraryCreateInfoKHR::pLibraries</code> with the pipeline
libraries to be used as inputs to linking, and set <code>VkPipelineLibraryCreateInfoKHR::libraryCount</code>
to the appropriate value</p>
</li>
</ul>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_KHR_deferred_host_operations">5. VK_KHR_deferred_host_operations</h2>
<div class="sectionbody">
<div class="paragraph">
<p><code>VK_KHR_deferred_host_operations</code> introduces a mechanism for distributing expensive
CPU tasks across multiple threads. Rather than introduce a thread pool into Vulkan
drivers, <code>VK_KHR_deferred_host_operations</code> is designed to allow an application to
create and manage the threads.</p>
</div>
<div class="paragraph">
<p>As with <code>VK_KHR_pipeline_library</code>, <code>VK_KHR_deferred_host_operations</code> was defined
as a separate extension to allow for the possibility of using the same functionality
in other extensions in the future without introducing a dependency on the ray
tracing extensions.</p>
</div>
<div class="paragraph">
<p>Only operations that are specifically noted as supporting deferral may be deferred.
Currently the only operations which support deferral are <code>vkCreateRayTracingPipelinesKHR</code>,
<code>vkBuildAccelerationStructuresKHR</code>, <code>vkCopyAccelerationStructureKHR</code>,
<code>vkCopyMemoryToAccelerationStructureKHR</code>, and <code>vkCopyAccelerationStructureToMemoryKHR</code></p>
</div>
<div class="paragraph">
<p>To request that an operation is deferred:</p>
</div>
<div class="ulist">
<ul>
<li>
<p>Create a <code>VkDeferredOperationKHR</code> object by calling <code>vkCreateDeferredOperationKHR</code></p>
</li>
<li>
<p>Call the operation that you wish to be deferred, passing the <code>VkDeferredOperationKHR</code>
as a parameter.</p>
</li>
<li>
<p>Check the <code>VkResult</code> returned by the above operation:</p>
<div class="ulist">
<ul>
<li>
<p><code>VK_OPERATION_DEFERRED_KHR</code> indicates that the operation was successfully
deferred</p>
</li>
<li>
<p><code>VK_OPERATION_NOT_DEFERRED_KHR</code> indicates that the operation successfully
completed immediately</p>
</li>
<li>
<p>Any error value indicates that an error occurred</p>
</li>
</ul>
</div>
</li>
</ul>
</div>
<div class="paragraph">
<p>To join a thread to a deferred operation, and contribute CPU time to progressing
the operation:</p>
</div>
<div class="ulist">
<ul>
<li>
<p>Call <code>vkDeferredOperationJoinKHR</code> from each thread that you wish to participate
in the operation</p>
</li>
<li>
<p>Check the <code>VkResult</code> returned by <code>vkDeferredOperationJoinKHR</code>:</p>
<div class="ulist">
<ul>
<li>
<p><code>VK_SUCCESS</code> indicates that the operation is complete</p>
</li>
<li>
<p><code>VK_THREAD_DONE_KHR</code> indicates that there is no more work to assign to the
calling thread, but that other threads may still have some additional work to
complete. The current thread should not attempt to re-join by calling
<code>vkDeferredOperationJoinKHR</code> again</p>
</li>
<li>
<p><code>VK_THREAD_IDLE_KHR</code> indicates that there is <strong>temporarily</strong> no work to assign
to the calling thread, but that additional work may become available in the
future. The current thread may perform some other useful work on the calling
thread, and re-joining by calling <code>vkDeferredOperationJoinKHR</code> again later
may prove beneficial</p>
</li>
</ul>
</div>
</li>
</ul>
</div>
<div class="paragraph">
<p>After an operation has completed (i.e. <code>vkDeferredOperationJoinKHR</code> has returned
<code>VK_SUCCESS</code>), call <code>vkGetDeferredOperationResultKHR</code> to get the result of the
operation.</p>
</div>
</div>
</div>