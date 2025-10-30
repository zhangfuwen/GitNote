<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#vulkan-release-summary">Vulkan Release Summary</a>
<ul class="sectlevel1">
<li><a href="#_vulkan_1_1">1. Vulkan 1.1</a></li>
<li><a href="#_vulkan_1_2">2. Vulkan 1.2</a></li>
<li><a href="#_vulkan_1_3">3. Vulkan 1.3</a></li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink:/Notes/004-3d-rendering/vulkan/chapters/vulkan_release_summary.html
layout: default
---</p>
</div>
<h1 id="vulkan-release-summary" class="sect0">Vulkan Release Summary</h1>
<div class="paragraph">
<p>Each minor release version of Vulkan <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#extendingvulkan-compatibility-promotion">promoted</a> a different set of extension to core. This means that it&#8217;s no longer necessary to enable an extensions to use it&#8217;s functionality if the application requests at least that Vulkan version (given that the version is supported by the implementation).</p>
</div>
<div class="paragraph">
<p>The following summary contains a list of the extensions added to the respective core versions and why they were added. This list is taken from the Vulkan spec, but links jump to the various spots in the Vulkan Guide</p>
</div>
<div class="sect1">
<h2 id="_vulkan_1_1">1. Vulkan 1.1</h2>
<div class="sectionbody">
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#versions-1.1">Vulkan Spec Section</a></p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>Vulkan 1.1 was released on March 7, 2018</p>
</div>
<div class="paragraph">
<p>Besides the listed extensions below, Vulkan 1.1 introduced the <a href="subgroups.html#subgroups">subgroups</a>, <a href="protected.html#protected">protected memory</a>, and the ability to query the instance version.</p>
</div>
<div class="ulist">
<ul>
<li>
<p><a href="extensions/shader_features.html#VK_KHR_16bit_storage">VK_KHR_16bit_storage</a></p>
</li>
<li>
<p><a href="extensions/cleanup.html#pnext-expansions">VK_KHR_bind_memory2</a></p>
</li>
<li>
<p><a href="extensions/cleanup.html#VK_KHR_dedicated_allocation">VK_KHR_dedicated_allocation</a></p>
</li>
<li>
<p><a href="extensions/VK_KHR_descriptor_update_template.html#VK_KHR_descriptor_update_template">VK_KHR_descriptor_update_template</a></p>
</li>
<li>
<p><a href="extensions/device_groups.html#device-groups">VK_KHR_device_group</a></p>
</li>
<li>
<p><a href="extensions/device_groups.html#device-groups">VK_KHR_device_group_creation</a></p>
</li>
<li>
<p><a href="extensions/external.html#external-memory">VK_KHR_external_fence</a></p>
</li>
<li>
<p><a href="extensions/external.html#external-memory">VK_KHR_external_fence_capabilities</a></p>
</li>
<li>
<p><a href="extensions/external.html#external-memory">VK_KHR_external_memory</a></p>
</li>
<li>
<p><a href="extensions/external.html#external-memory">VK_KHR_external_memory_capabilities</a></p>
</li>
<li>
<p><a href="extensions/external.html#external-memory">VK_KHR_external_semaphore</a></p>
</li>
<li>
<p><a href="extensions/external.html#external-memory">VK_KHR_external_semaphore_capabilities</a></p>
</li>
<li>
<p><a href="extensions/cleanup.html#pnext-expansions">VK_KHR_get_memory_requirements2</a></p>
</li>
<li>
<p><a href="extensions/cleanup.html#pnext-expansions">VK_KHR_get_physical_device_properties2</a></p>
</li>
<li>
<p><a href="extensions/cleanup.html#maintenance-extensions">VK_KHR_maintenance1</a></p>
</li>
<li>
<p><a href="extensions/cleanup.html#maintenance-extensions">VK_KHR_maintenance2</a></p>
</li>
<li>
<p><a href="extensions/cleanup.html#maintenance-extensions">VK_KHR_maintenance3</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_KHR_multiview.html#_description">VK_KHR_multiview</a></p>
</li>
<li>
<p><a href="shader_memory_layout.html#VK_KHR_relaxed_block_layout">VK_KHR_relaxed_block_layout</a></p>
</li>
<li>
<p><a href="extensions/VK_KHR_sampler_ycbcr_conversion.html#VK_KHR_sampler_ycbcr_conversion">VK_KHR_sampler_ycbcr_conversion</a></p>
</li>
<li>
<p><a href="extensions/shader_features.html#VK_KHR_shader_draw_parameters">VK_KHR_shader_draw_parameters</a></p>
</li>
<li>
<p><a href="extensions/shader_features.html#VK_KHR_storage_buffer_storage_class">VK_KHR_storage_buffer_storage_class</a></p>
</li>
<li>
<p><a href="extensions/shader_features.html#VK_KHR_variable_pointers">VK_KHR_variable_pointers</a></p>
</li>
</ul>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_vulkan_1_2">2. Vulkan 1.2</h2>
<div class="sectionbody">
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#versions-1.2">Vulkan Spec Section</a></p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>Vulkan 1.2 was released on January 15, 2020</p>
</div>
<div class="ulist">
<ul>
<li>
<p><a href="extensions/shader_features.html#VK_KHR_8bit_storage">VK_KHR_8bit_storage</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_KHR_buffer_device_address.html#_description">VK_KHR_buffer_device_address</a></p>
</li>
<li>
<p><a href="extensions/cleanup.html#pnext-expansions">VK_KHR_create_renderpass2</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_KHR_depth_stencil_resolve.html#_description">VK_KHR_depth_stencil_resolve</a></p>
</li>
<li>
<p><a href="extensions/VK_KHR_draw_indirect_count.html#VK_KHR_draw_indirect_count">VK_KHR_draw_indirect_count</a></p>
</li>
<li>
<p><a href="extensions/cleanup.html#VK_KHR_driver_properties">VK_KHR_driver_properties</a></p>
</li>
<li>
<p><a href="extensions/VK_KHR_image_format_list.html#VK_KHR_image_format_list">VK_KHR_image_format_list</a></p>
</li>
<li>
<p><a href="extensions/VK_KHR_imageless_framebuffer.html#VK_KHR_imageless_framebuffer">VK_KHR_imageless_framebuffer</a></p>
</li>
<li>
<p><a href="extensions/cleanup.html#VK_KHR_sampler_mirror_clamp_to_edge">VK_KHR_sampler_mirror_clamp_to_edge</a></p>
</li>
<li>
<p><a href="extensions/cleanup.html#VK_KHR_separate_depth_stencil_layouts">VK_KHR_separate_depth_stencil_layouts</a></p>
</li>
<li>
<p><a href="atomics.html#VK_KHR_shader_atomic_int64">VK_KHR_shader_atomic_int64</a></p>
</li>
<li>
<p><a href="extensions/shader_features.html#VK_KHR_shader_float16_int8">VK_KHR_shader_float16_int8</a></p>
</li>
<li>
<p><a href="extensions/shader_features.html#VK_KHR_shader_float_controls">VK_KHR_shader_float_controls</a></p>
</li>
<li>
<p><a href="subgroups.html#VK_KHR_shader_subgroup_extended_types">VK_KHR_shader_subgroup_extended_types</a></p>
</li>
<li>
<p><a href="extensions/shader_features.html#VK_KHR_spirv_1_4">VK_KHR_spirv_1_4</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/blog/vulkan-timeline-semaphores">VK_KHR_timeline_semaphore</a></p>
</li>
<li>
<p><a href="shader_memory_layout.html#VK_KHR_uniform_buffer_standard_layout">VK_KHR_uniform_buffer_standard_layout</a></p>
</li>
<li>
<p><a href="extensions/shader_features.html#VK_KHR_vulkan_memory_model">VK_KHR_vulkan_memory_model</a></p>
</li>
<li>
<p><a href="extensions/VK_EXT_descriptor_indexing.html#VK_EXT_descriptor_indexing">VK_EXT_descriptor_indexing</a></p>
</li>
<li>
<p><a href="extensions/cleanup.html#VK_EXT_host_query_reset">VK_EXT_host_query_reset</a></p>
</li>
<li>
<p><a href="extensions/cleanup.html#VK_EXT_sampler_filter_minmax">VK_EXT_sampler_filter_minmax</a></p>
</li>
<li>
<p><a href="shader_memory_layout.html#VK_EXT_scalar_block_layout">VK_EXT_scalar_block_layout</a></p>
</li>
<li>
<p><a href="extensions/cleanup.html#VK_EXT_separate_stencil_usage">VK_EXT_separate_stencil_usage</a></p>
</li>
<li>
<p><a href="extensions/shader_features.html#VK_EXT_shader_viewport_index_layer">VK_EXT_shader_viewport_index_layer</a></p>
</li>
</ul>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_vulkan_1_3">3. Vulkan 1.3</h2>
<div class="sectionbody">
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#versions-1.3">Vulkan Spec Section</a></p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>Vulkan 1.3 was released on January 25, 2022</p>
</div>
<div class="ulist">
<ul>
<li>
<p><a href="extensions/cleanup.html#pnext-expansions">VK_KHR_copy_commands2</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/blog/streamlining-render-passes">VK_KHR_dynamic_rendering</a></p>
</li>
<li>
<p><a href="extensions/cleanup.html#VK_KHR_format_feature_flags2">VK_KHR_format_feature_flags2</a></p>
</li>
<li>
<p><a href="extensions/cleanup.html#VK_KHR_maintenance4">VK_KHR_maintenance4</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_KHR_shader_integer_dot_product.html#_description">VK_KHR_shader_integer_dot_product</a></p>
</li>
<li>
<p><a href="extensions/shader_features.html#VK_KHR_shader_non_semantic_info">VK_KHR_shader_non_semantic_info</a></p>
</li>
<li>
<p><a href="extensions/shader_features.html#VK_KHR_shader_terminate_invocation">VK_KHR_shader_terminate_invocation</a></p>
</li>
<li>
<p><a href="extensions/VK_KHR_synchronization2.html">VK_KHR_synchronization2</a></p>
</li>
<li>
<p><a href="extensions/shader_features.html#VK_KHR_zero_initialize_workgroup_memory">VK_KHR_zero_initialize_workgroup_memory</a></p>
</li>
<li>
<p><a href="extensions/cleanup.html#VK_EXT_4444_formats-and-VK_EXT_ycbcr_2plane_444_formats">VK_EXT_4444_formats</a></p>
</li>
<li>
<p><a href="dynamic_state.html#states-that-are-dynamic">VK_EXT_extended_dynamic_state</a></p>
</li>
<li>
<p><a href="dynamic_state.html#states-that-are-dynamic">VK_EXT_extended_dynamic_state2</a></p>
</li>
<li>
<p><a href="extensions/VK_EXT_inline_uniform_block.html#VK_EXT_inline_uniform_block">VK_EXT_inline_uniform_block</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_EXT_pipeline_creation_cache_control.html#_description">VK_EXT_pipeline_creation_cache_control</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_EXT_pipeline_creation_feedback.html#_description">VK_EXT_pipeline_creation_feedback</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_EXT_private_data.html#_description">VK_EXT_private_data</a></p>
</li>
<li>
<p><a href="extensions/shader_features.html#VK_EXT_shader_demote_to_helper_invocation">VK_EXT_shader_demote_to_helper_invocation</a></p>
</li>
<li>
<p><a href="subgroups.html#VK_EXT_subgroup_size_control">VK_EXT_subgroup_size_control</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_EXT_texel_buffer_alignment.html#_description">VK_EXT_texel_buffer_alignment</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_EXT_texture_compression_astc_hdr.html#_description">VK_EXT_texture_compression_astc_hdr</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_EXT_tooling_info.html#_description">VK_EXT_tooling_info</a></p>
</li>
<li>
<p><a href="extensions/cleanup.html#VK_EXT_4444_formats-and-VK_EXT_ycbcr_2plane_444_formats">VK_EXT_ycbcr_2plane_444_formats</a></p>
</li>
</ul>
</div>
</div>
</div>