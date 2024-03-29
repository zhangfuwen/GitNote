# ANDROID_extension_pack_es31a

Name

    ANDROID_extension_pack_es31a

Name Strings

    GL_ANDROID_extension_pack_es31a

Contact

    Jesse Hall (jessehall 'at' google.com)

Contributors

    Jesse Hall, Google

Status

    Complete.

Version

    Last Modified Date: July 7, 2014
    Revision: 3

Number

    OpenGL ES Extension #187

Dependencies

    OpenGL ES 3.1 and GLSL ES 3.10 are required.

    The following extensions are required:
      * KHR_debug
      * KHR_texture_compression_astc_ldr
      * KHR_blend_equation_advanced
      * OES_sample_shading
      * OES_sample_variables
      * OES_shader_image_atomic
      * OES_shader_multisample_interpolation
      * OES_texture_stencil8
      * OES_texture_storage_multisample_2d_array
      * EXT_copy_image
      * EXT_draw_buffers_indexed
      * EXT_geometry_shader
      * EXT_gpu_shader5
      * EXT_primitive_bounding_box
      * EXT_shader_io_blocks
      * EXT_tessellation_shader
      * EXT_texture_border_clamp
      * EXT_texture_buffer
      * EXT_texture_cube_map_array
      * EXT_texture_sRGB_decode

Overview

    This extension changes little functionality directly. Instead it serves to
    roll up the 20 extensions it requires, allowing applications to check for
    all of them at once, and enable all of their shading language features with
    a single #extension statement. The Android platform provides special support
    outside of OpenGL ES to help applications target this set of extensions.

    In addition, this extension ensures support for images, shader storage
    buffers, and atomic counters in fragment shaders. In unextended OpenGL ES
    the minimum value of the relevant implementation-defined limits is zero;
    this extension raises these minimums to match the minimums for compute
    shaders.

New Procedures and Functions

    None

New Tokens

    None

Modifications to Chapter 20 of the OpenGL ES 3.1 Specification (State Tables)

    Modify Table 20.44, Implementation Dependent Fragment Shader Limits, p. 395:

    Get Value                                      Type  Get Command    Minimum Value  Description                Sec. 
    -----------------------                        ----  -----------    -------------  -------------------------  -----
    MAX_FRAGMENT_ATOMIC_COUNTER_BUFFERS            Z+    GetIntegerv    1              No. of atomic counter      7.7
                                                                                       buffers accessed by a
                                                                                       fragment shader
    MAX_FRAGMENT_ATOMIC_COUNTERS                   Z+    GetIntegerv    8              No. of atomic counters     7.7
                                                                                       accessed by a fragment
                                                                                       shader
    MAX_FRAGMENT_IMAGE_UNIFORMS                    Z+    GetIntegerv    4              No. of image variables in  11.1.3
                                                                                       fragment shaders
    MAX_FRAGMENT_SHADER_STORAGE_BLOCKS             Z+    GetIntegerv    4              No. of shader storage      7.8
                                                                                       blocks accessed by a 
                                                                                       fragment shader
    [[Change minimum values]]

Modifications to The OpenGL ES Shading Language Specification, Version 3.10
(Revision 5)

    Including the following line in a shader:

      #extension GL_ANDROID_extension_pack_es31a : <behavior>

    has the same effect as including the following lines:

      #extension GL_KHR_blend_equation_advanced : <behavior>
      #extension GL_OES_sample_variables : <behavior>
      #extension GL_OES_shader_image_atomic : <behavior>
      #extension GL_OES_shader_multisample_interpolation : <behavior>
      #extension GL_OES_texture_storage_multisample_2d_array : <behavior>
      #extension GL_EXT_geometry_shader : <behavior>
      #extension GL_EXT_gpu_shader5 : <behavior>
      #extension GL_EXT_primitive_bounding_box : <behavior>
      #extension GL_EXT_shader_io_blocks : <behavior>
      #extension GL_EXT_tessellation_shader : <behavior>
      #extension GL_EXT_texture_buffer : <behavior>
      #extension GL_EXT_texture_cube_map_array : <behavior>

    where <behavior> is as specified in section 3.4.

    A new preprocessor #define is added to the OpenGL ES Shading Language:

      #define GL_ANDROID_extension_pack_es31a   1

New Implementation Dependent State

    None

Issues

    (1) Should all the extensions be enabled in all shader stages? What happens
    if an extension is enabled in a stage that it does not modify?

    RESOLVED: All extensions are enabled in all stages.

    Enabling this extension in a shader has exactly the same behavior as
    enabling each of the required extensions individually. None of the other
    extensions limit what shader stages they can be enabled in. Generally, if
    the modifications introduced by an extension do not apply to a shader stage,
    then enabling it in that stage is a no-op. If an extension does introduce
    new features to the language in a stage, a shader that doesn't use the new
    features will behave the same whether it enables the extension or not.


Revision History

    Revision 3, 2014/7/7 (Jesse Hall)
        - Changed status from "draft" to "complete"
        - Corrected capitalization of EXT_texture_sRGB_decode

    Revision 2, 2014/5/2 (Jesse Hall)
        - Added dependency on EXT_primitive_bounding_box
        - Added and resolved Issue #1

    Revision 1, 2014/4/18 (Jesse Hall)
        - Initial version
