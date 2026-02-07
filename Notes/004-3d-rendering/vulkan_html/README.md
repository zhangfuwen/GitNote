<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#_vulkan_guide">Vulkan<sup>®</sup> Guide</a>
<ul class="sectlevel1">
<li><a href="#_logistics_overview">1. Logistics Overview</a>
<ul class="sectlevel2">
<li><a href="#_what_is_vulkan">1.1. What is Vulkan?</a></li>
<li><a href="#_what_you_can_do_with_vulkan">1.2. What you can do with Vulkan</a></li>
<li><a href="#_vulkan_spec">1.3. Vulkan Spec</a></li>
<li><a href="#_platforms">1.4. Platforms</a></li>
<li><a href="#_checking_for_support">1.5. Checking for Support</a></li>
<li><a href="#_versions">1.6. Versions</a></li>
<li><a href="#_vulkan_release_summary">1.7. Vulkan Release Summary</a></li>
<li><a href="#_what_is_spir_v">1.8. What is SPIR-V?</a></li>
<li><a href="#_portability_initiative">1.9. Portability Initiative</a></li>
<li><a href="#_vulkan_cts">1.10. Vulkan CTS</a></li>
<li><a href="#_vulkan_development_tools">1.11. Vulkan Development Tools</a></li>
<li><a href="#_vulkan_validation_overview">1.12. Vulkan Validation Overview</a></li>
<li><a href="#_vulkan_decoder_ring_gl_gles_directx_and_metal">1.13. Vulkan Decoder Ring (GL, GLES, DirectX, and Metal)</a></li>
</ul>
</li>
<li><a href="#_using_vulkan">2. Using Vulkan</a>
<ul class="sectlevel2">
<li><a href="#_loader">2.1. Loader</a></li>
<li><a href="#_layers">2.2. Layers</a></li>
<li><a href="#_querying_properties_extensions_features_limits_and_formats">2.3. Querying Properties, Extensions, Features, Limits, and Formats</a></li>
<li><a href="#_queues_and_queue_family">2.4. Queues and Queue Family</a></li>
<li><a href="#_wsi">2.5. WSI</a></li>
<li><a href="#_pnext_and_stype">2.6. pNext and sType</a></li>
<li><a href="#_synchronization">2.7. Synchronization</a></li>
<li><a href="#_memory_allocation_strategy">2.8. Memory Allocation Strategy</a></li>
<li><a href="#_pipeline_cachingderivatives">2.9. Pipeline Caching/Derivatives</a></li>
<li><a href="#_threading">2.10. Threading</a></li>
<li><a href="#_depth">2.11. Depth</a></li>
<li><a href="#_mapping_data_to_shaders">2.12. Mapping Data to Shaders</a></li>
<li><a href="#_robustness">2.13. Robustness</a></li>
<li><a href="#_dynamic_state">2.14. Dynamic State</a></li>
<li><a href="#_subgroups">2.15. Subgroups</a></li>
<li><a href="#_shader_memory_layout">2.16. Shader Memory Layout</a></li>
<li><a href="#_atomics">2.17. Atomics</a></li>
<li><a href="#_common_pitfalls">2.18. Common Pitfalls</a></li>
<li><a href="#_using_hlsl_shaders">2.19. Using HLSL shaders</a></li>
</ul>
</li>
<li><a href="#_when_and_why_to_use_extensions">3. When and Why to use Extensions</a>
<ul class="sectlevel2">
<li><a href="#_cleanup_extensions">3.1. Cleanup Extensions</a></li>
<li><a href="#_device_groups">3.2. Device Groups</a></li>
<li><a href="#_external_memory_and_sychronization">3.3. External Memory and Sychronization</a></li>
<li><a href="#_ray_tracing">3.4. Ray Tracing</a></li>
<li><a href="#_shader_features">3.5. Shader Features</a></li>
<li><a href="#_translation_layer_extensions">3.6. Translation Layer Extensions</a></li>
<li><a href="#_vk_ext_descriptor_indexing">3.7. VK_EXT_descriptor_indexing</a></li>
<li><a href="#_vk_ext_inline_uniform_block">3.8. VK_EXT_inline_uniform_block</a></li>
<li><a href="#_vk_ext_memory_priority">3.9. VK_EXT_memory_priority</a></li>
<li><a href="#_vk_khr_descriptor_update_template">3.10. VK_KHR_descriptor_update_template</a></li>
<li><a href="#_vk_khr_draw_indirect_count">3.11. VK_KHR_draw_indirect_count</a></li>
<li><a href="#_vk_khr_image_format_list">3.12. VK_KHR_image_format_list</a></li>
<li><a href="#_vk_khr_imageless_framebuffer">3.13. VK_KHR_imageless_framebuffer</a></li>
<li><a href="#_vk_khr_sampler_ycbcr_conversion">3.14. VK_KHR_sampler_ycbcr_conversion</a></li>
<li><a href="#_vk_khr_timeline_semaphore">3.15. VK_KHR_timeline_semaphore</a></li>
<li><a href="#_vk_khr_dynamic_rendering">3.16. VK_KHR_dynamic_rendering</a></li>
<li><a href="#_vk_khr_shader_subgroup_uniform_control_flow">3.17. VK_KHR_shader_subgroup_uniform_control_flow</a></li>
</ul>
</li>
<li><a href="#_contributing">4. Contributing</a></li>
<li><a href="#_license">5. License</a></li>
<li><a href="#_code_of_conduct">6. Code of conduct</a></li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>layout: default
permalink: /sample/
---</p>
</div>
<h1 id="_vulkan_guide" class="sect0">Vulkan<sup>®</sup> Guide</h1>
<div class="paragraph">
<p>The Khronos<sup>®</sup> Vulkan Working Group
:data-uri:
:icons: font
:max-width: 100%
:numbered:
:source-highlighter: rouge
:rouge-style: github</p>
</div>
<div class="imageblock">
<div class="content">
<img src="images/vulkan_logo.png" alt="Vulkan Logo">
</div>
</div>
<div class="imageblock">
<div class="content">
<img src="images/khronos_logo.png" alt="Khronos logo">
</div>
</div>
<div class="paragraph">
<p>The Vulkan Guide is designed to help developers get up and going with the world of Vulkan. It is aimed to be a light read that leads to many other useful links depending on what a developer is looking for. All information is intended to help better fill the gaps about the many nuances of Vulkan.</p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>The Vulkan Guide can be built as a single page using <code>asciidoctor guide.adoc</code></p>
</div>
</td>
</tr>
</table>
</div>
<div class="sect1">
<h2 id="_logistics_overview">1. Logistics Overview</h2>
<div class="sectionbody">
<div class="sect2">
<h3 id="_what_is_vulkan">1.1. <a href="chapters/what_is_vulkan.html">What is Vulkan?</a></h3>

</div>
<div class="sect2">
<h3 id="_what_you_can_do_with_vulkan">1.2. <a href="chapters/what_vulkan_can_do.html">What you can do with Vulkan</a></h3>

</div>
<div class="sect2">
<h3 id="_vulkan_spec">1.3. <a href="chapters/vulkan_spec.html">Vulkan Spec</a></h3>

</div>
<div class="sect2">
<h3 id="_platforms">1.4. <a href="chapters/platforms.html">Platforms</a></h3>

</div>
<div class="sect2">
<h3 id="_checking_for_support">1.5. <a href="chapters/checking_for_support.html">Checking for Support</a></h3>

</div>
<div class="sect2">
<h3 id="_versions">1.6. <a href="chapters/versions.html">Versions</a></h3>

</div>
<div class="sect2">
<h3 id="_vulkan_release_summary">1.7. <a href="chapters/vulkan_release_summary.html">Vulkan Release Summary</a></h3>

</div>
<div class="sect2">
<h3 id="_what_is_spir_v">1.8. <a href="chapters/what_is_spirv.html">What is SPIR-V?</a></h3>

</div>
<div class="sect2">
<h3 id="_portability_initiative">1.9. <a href="chapters/portability_initiative.html">Portability Initiative</a></h3>

</div>
<div class="sect2">
<h3 id="_vulkan_cts">1.10. <a href="chapters/vulkan_cts.html">Vulkan CTS</a></h3>

</div>
<div class="sect2">
<h3 id="_vulkan_development_tools">1.11. <a href="chapters/development_tools.html">Vulkan Development Tools</a></h3>

</div>
<div class="sect2">
<h3 id="_vulkan_validation_overview">1.12. <a href="chapters/validation_overview.html">Vulkan Validation Overview</a></h3>

</div>
<div class="sect2">
<h3 id="_vulkan_decoder_ring_gl_gles_directx_and_metal">1.13. <a href="chapters/decoder_ring.html">Vulkan Decoder Ring (GL, GLES, DirectX, and Metal)</a></h3>

</div>
</div>
</div>
<div class="sect1">
<h2 id="_using_vulkan">2. Using Vulkan</h2>
<div class="sectionbody">
<div class="sect2">
<h3 id="_loader">2.1. <a href="chapters/loader.html">Loader</a></h3>

</div>
<div class="sect2">
<h3 id="_layers">2.2. <a href="chapters/layers.html">Layers</a></h3>

</div>
<div class="sect2">
<h3 id="_querying_properties_extensions_features_limits_and_formats">2.3. <a href="chapters/querying_extensions_features.html">Querying Properties, Extensions, Features, Limits, and Formats</a></h3>
<div class="sect3">
<h4 id="_enabling_vulkan_extensions">2.3.1. <a href="chapters/enabling_extensions.html">Enabling Vulkan Extensions</a></h4>

</div>
<div class="sect3">
<h4 id="_enabling_vulkan_features">2.3.2. <a href="chapters/enabling_features.html">Enabling Vulkan Features</a></h4>

</div>
<div class="sect3">
<h4 id="_using_spir_v_extension">2.3.3. <a href="chapters/spirv_extensions.html">Using SPIR-V Extension</a></h4>

</div>
<div class="sect3">
<h4 id="_formats">2.3.4. <a href="chapters/formats.html">Formats</a></h4>

</div>
</div>
<div class="sect2">
<h3 id="_queues_and_queue_family">2.4. <a href="chapters/queues.html">Queues and Queue Family</a></h3>

</div>
<div class="sect2">
<h3 id="_wsi">2.5. <a href="chapters/wsi.html">WSI</a></h3>

</div>
<div class="sect2">
<h3 id="_pnext_and_stype">2.6. <a href="chapters/pnext_and_stype.html">pNext and sType</a></h3>

</div>
<div class="sect2">
<h3 id="_synchronization">2.7. <a href="chapters/synchronization.html">Synchronization</a></h3>
<div class="sect3">
<h4 id="_porting_to_vk_khr_synchronization2">2.7.1. <a href="chapters/extensions/VK_KHR_synchronization2.html">Porting to VK_KHR_synchronization2</a></h4>

</div>
</div>
<div class="sect2">
<h3 id="_memory_allocation_strategy">2.8. <a href="chapters/memory_allocation.html">Memory Allocation Strategy</a></h3>
<div class="sect3">
<h4 id="_sparse_resources">2.8.1. <a href="chapters/sparse_resources.html">Sparse Resources</a></h4>

</div>
<div class="sect3">
<h4 id="_protected_memory">2.8.2. <a href="chapters/protected.html">Protected Memory</a></h4>

</div>
</div>
<div class="sect2">
<h3 id="_pipeline_cachingderivatives">2.9. <a href="chapters/pipeline_cache.html">Pipeline Caching/Derivatives</a></h3>

</div>
<div class="sect2">
<h3 id="_threading">2.10. <a href="chapters/threading.html">Threading</a></h3>

</div>
<div class="sect2">
<h3 id="_depth">2.11. <a href="chapters/depth.html">Depth</a></h3>
<div class="ulist">
<ul>
<li>
<p><code>VK_EXT_depth_range_unrestricted</code>, <code>VK_EXT_depth_clip_enable</code>, <code>VK_EXT_depth_clip_control</code></p>
</li>
</ul>
</div>
</div>
<div class="sect2">
<h3 id="_mapping_data_to_shaders">2.12. <a href="chapters/mapping_data_to_shaders.html">Mapping Data to Shaders</a></h3>
<div class="sect3">
<h4 id="_vertex_input_data_processing">2.12.1. <a href="chapters/vertex_input_data_processing.html">Vertex Input Data Processing</a></h4>

</div>
<div class="sect3">
<h4 id="_descriptor_dynamic_offset">2.12.2. <a href="chapters/descriptor_dynamic_offset.html">Descriptor Dynamic Offset</a></h4>

</div>
</div>
<div class="sect2">
<h3 id="_robustness">2.13. <a href="chapters/robustness.html">Robustness</a></h3>
<div class="ulist">
<ul>
<li>
<p><code>VK_EXT_image_robustness</code>, <code>VK_EXT_robustness2</code></p>
</li>
</ul>
</div>
</div>
<div class="sect2">
<h3 id="_dynamic_state">2.14. <a href="chapters/dynamic_state.html">Dynamic State</a></h3>
<div class="ulist">
<ul>
<li>
<p><code>VK_EXT_extended_dynamic_state</code>, <code>VK_EXT_extended_dynamic_state2</code>, <code>VK_EXT_vertex_input_dynamic_state</code>, <code>VK_EXT_color_write_enable</code></p>
</li>
</ul>
</div>
</div>
<div class="sect2">
<h3 id="_subgroups">2.15. <a href="chapters/subgroups.html">Subgroups</a></h3>
<div class="ulist">
<ul>
<li>
<p><code>VK_EXT_subgroup_size_control</code>, <code>VK_KHR_shader_subgroup_extended_types</code>, <code>VK_EXT_shader_subgroup_ballot</code>, <code>VK_EXT_shader_subgroup_vote</code></p>
</li>
</ul>
</div>
</div>
<div class="sect2">
<h3 id="_shader_memory_layout">2.16. <a href="chapters/shader_memory_layout.html">Shader Memory Layout</a></h3>
<div class="ulist">
<ul>
<li>
<p><code>VK_KHR_uniform_buffer_standard_layout</code>, <code>VK_KHR_relaxed_block_layout</code>, <code>VK_EXT_scalar_block_layout</code></p>
</li>
</ul>
</div>
</div>
<div class="sect2">
<h3 id="_atomics">2.17. <a href="chapters/atomics.html">Atomics</a></h3>
<div class="ulist">
<ul>
<li>
<p><code>VK_KHR_shader_atomic_int64</code>, <code>VK_EXT_shader_image_atomic_int64</code>, <code>VK_EXT_shader_atomic_float</code>, <code>VK_EXT_shader_atomic_float2</code></p>
</li>
</ul>
</div>
</div>
<div class="sect2">
<h3 id="_common_pitfalls">2.18. <a href="chapters/common_pitfalls.html">Common Pitfalls</a></h3>

</div>
<div class="sect2">
<h3 id="_using_hlsl_shaders">2.19. <a href="chapters/hlsl.html">Using HLSL shaders</a></h3>

</div>
</div>
</div>
<div class="sect1">
<h2 id="_when_and_why_to_use_extensions">3. When and Why to use Extensions</h2>
<div class="sectionbody">
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>These are supplemental references for the various Vulkan Extensions. Please consult the Vulkan Spec for further details on any extension</p>
</div>
</td>
</tr>
</table>
</div>
<div class="sect2">
<h3 id="_cleanup_extensions">3.1. <a href="chapters/extensions/cleanup.html">Cleanup Extensions</a></h3>
<div class="ulist">
<ul>
<li>
<p><code>VK_EXT_4444_formats</code>, <code>VK_KHR_bind_memory2</code>, <code>VK_KHR_create_renderpass2</code>, <code>VK_KHR_dedicated_allocation</code>, <code>VK_KHR_driver_properties</code>, <code>VK_KHR_get_memory_requirements2</code>, <code>VK_KHR_get_physical_device_properties2</code>, <code>VK_EXT_host_query_reset</code>, <code>VK_KHR_maintenance1</code>, <code>VK_KHR_maintenance2</code>, <code>VK_KHR_maintenance3</code>, <code>VK_KHR_maintenance4</code>, <code>VK_KHR_separate_depth_stencil_layouts</code>, <code>VK_KHR_depth_stencil_resolve</code>, <code>VK_EXT_separate_stencil_usage</code>, <code>VK_EXT_sampler_filter_minmax</code>, <code>VK_KHR_sampler_mirror_clamp_to_edge</code>, <code>VK_EXT_ycbcr_2plane_444_formats</code>, <code>VK_KHR_format_feature_flags2</code>, <code>VK_EXT_rgba10x6_formats</code>, <code>VK_KHR_copy_commands2</code></p>
</li>
</ul>
</div>
</div>
<div class="sect2">
<h3 id="_device_groups">3.2. <a href="chapters/extensions/device_groups.html">Device Groups</a></h3>
<div class="ulist">
<ul>
<li>
<p><code>VK_KHR_device_group</code>, <code>VK_KHR_device_group_creation</code></p>
</li>
</ul>
</div>
</div>
<div class="sect2">
<h3 id="_external_memory_and_sychronization">3.3. <a href="chapters/extensions/external.html">External Memory and Sychronization</a></h3>
<div class="ulist">
<ul>
<li>
<p><code>VK_KHR_external_fence</code>, <code>VK_KHR_external_memory</code>, <code>VK_KHR_external_semaphore</code></p>
</li>
</ul>
</div>
</div>
<div class="sect2">
<h3 id="_ray_tracing">3.4. <a href="chapters/extensions/ray_tracing.html">Ray Tracing</a></h3>
<div class="ulist">
<ul>
<li>
<p><code>VK_KHR_acceleration_structure</code>, <code>VK_KHR_ray_tracing_pipeline</code>, <code>VK_KHR_ray_query</code>, <code>VK_KHR_pipeline_library</code>, <code>VK_KHR_deferred_host_operations</code></p>
</li>
</ul>
</div>
</div>
<div class="sect2">
<h3 id="_shader_features">3.5. <a href="chapters/extensions/shader_features.html">Shader Features</a></h3>
<div class="ulist">
<ul>
<li>
<p><code>VK_KHR_8bit_storage</code>, <code>VK_KHR_16bit_storage</code>, <code>VK_KHR_shader_clock</code>, <code>VK_EXT_shader_demote_to_helper_invocation</code>, <code>VK_KHR_shader_draw_parameters</code>, <code>VK_KHR_shader_float16_int8</code>, <code>VK_KHR_shader_float_controls</code>, <code>VK_KHR_shader_non_semantic_info</code>, <code>VK_EXT_shader_stencil_export</code>, <code>VK_KHR_shader_terminate_invocation</code>, <code>VK_EXT_shader_viewport_index_layer</code>, <code>VK_KHR_spirv_1_4</code>, <code>VK_KHR_storage_buffer_storage_class</code>, <code>VK_KHR_variable_pointers</code>, <code>VK_KHR_vulkan_memory_model</code>, <code>VK_KHR_workgroup_memory_explicit_layout</code>, <code>VK_KHR_zero_initialize_workgroup_memory</code></p>
</li>
</ul>
</div>
</div>
<div class="sect2">
<h3 id="_translation_layer_extensions">3.6. <a href="chapters/extensions/translation_layer_extensions.html">Translation Layer Extensions</a></h3>
<div class="ulist">
<ul>
<li>
<p><code>VK_EXT_custom_border_color</code>, <code>VK_EXT_border_color_swizzle</code>, <code>VK_EXT_depth_clip_enable</code>, <code>VK_EXT_depth_clip_control</code>, <code>VK_EXT_provoking_vertex</code>, <code>VK_EXT_transform_feedback</code>, <code>VK_EXT_image_view_min_lod</code></p>
</li>
</ul>
</div>
</div>
<div class="sect2">
<h3 id="_vk_ext_descriptor_indexing">3.7. <a href="chapters/extensions/VK_EXT_descriptor_indexing.html">VK_EXT_descriptor_indexing</a></h3>

</div>
<div class="sect2">
<h3 id="_vk_ext_inline_uniform_block">3.8. <a href="chapters/extensions/VK_EXT_inline_uniform_block.html">VK_EXT_inline_uniform_block</a></h3>

</div>
<div class="sect2">
<h3 id="_vk_ext_memory_priority">3.9. <a href="chapters/extensions/VK_EXT_memory_priority.html">VK_EXT_memory_priority</a></h3>

</div>
<div class="sect2">
<h3 id="_vk_khr_descriptor_update_template">3.10. <a href="chapters/extensions/VK_KHR_descriptor_update_template.html">VK_KHR_descriptor_update_template</a></h3>

</div>
<div class="sect2">
<h3 id="_vk_khr_draw_indirect_count">3.11. <a href="chapters/extensions/VK_KHR_draw_indirect_count.html">VK_KHR_draw_indirect_count</a></h3>

</div>
<div class="sect2">
<h3 id="_vk_khr_image_format_list">3.12. <a href="chapters/extensions/VK_KHR_image_format_list.html">VK_KHR_image_format_list</a></h3>

</div>
<div class="sect2">
<h3 id="_vk_khr_imageless_framebuffer">3.13. <a href="chapters/extensions/VK_KHR_imageless_framebuffer.html">VK_KHR_imageless_framebuffer</a></h3>

</div>
<div class="sect2">
<h3 id="_vk_khr_sampler_ycbcr_conversion">3.14. <a href="chapters/extensions/VK_KHR_sampler_ycbcr_conversion.html">VK_KHR_sampler_ycbcr_conversion</a></h3>

</div>
<div class="sect2">
<h3 id="_vk_khr_timeline_semaphore">3.15. <a href="https://www.khronos.org/blog/vulkan-timeline-semaphores">VK_KHR_timeline_semaphore</a></h3>

</div>
<div class="sect2">
<h3 id="_vk_khr_dynamic_rendering">3.16. <a href="https://www.khronos.org/blog/streamlining-render-passes">VK_KHR_dynamic_rendering</a></h3>

</div>
<div class="sect2">
<h3 id="_vk_khr_shader_subgroup_uniform_control_flow">3.17. <a href="chapters/extensions/VK_KHR_shader_subgroup_uniform_control_flow.html">VK_KHR_shader_subgroup_uniform_control_flow</a></h3>

</div>
</div>
</div>
<div class="sect1">
<h2 id="_contributing">4. <a href="CONTRIBUTING.adoc">Contributing</a></h2>
<div class="sectionbody">

</div>
</div>
<div class="sect1">
<h2 id="_license">5. <a href="LICENSE">License</a></h2>
<div class="sectionbody">

</div>
</div>
<div class="sect1">
<h2 id="_code_of_conduct">6. <a href="CODE_OF_CONDUCT.adoc">Code of conduct</a></h2>
<div class="sectionbody">

</div>
</div>