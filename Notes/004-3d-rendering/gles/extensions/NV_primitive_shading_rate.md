# NV_primitive_shading_rate

Name

    NV_primitive_shading_rate

Name Strings

    GL_NV_primitive_shading_rate

Contact

    Pat Brown, NVIDIA Corporation (pbrown 'at' nvidia.com)

Contributors

    Jeff Bolz, NVIDIA

Status

    Shipping

Version

    Last Modified:      October 30, 2020
    Revision:           1

Number

    OpenGL Extension #554
    OpenGL ES Extension #332

Dependencies

    This extension is written against the OpenGL 4.6 Specification
    (Compatibility Profile), dated February 2, 2019.

    OpenGL 4.5 or OpenGL ES 3.2 is required.

    NV_shading_rate_image is required.

    This extension requires support from the OpenGL Shading Language (GLSL)
    extension "NV_primitive_shading_rate", which can be found at the Khronos
    Group Github site here:

        https://github.com/KhronosGroup/GLSL

    This extension interacts with NV_mesh_shader.

Overview

    This extension builds on top of the NV_shading_rate_image extension to
    provide OpenGL API support for using a per-primitive shading rate value to
    control the computation of the rate used to process each fragment.

    In the NV_shading_rate_image extension, the shading rate for each fragment
    produced by a primitive is determined by looking up a texel in the bound
    shading rate image and using that value as an index into a shading rate
    palette.  That extension provides a separate shading rate image lookup
    enable and palette for each viewport.  When a primitive is rasterized, the
    implementation uses the enable and palette associated with the primitive's
    viewport to determine the shading rate.

    This extension decouples the shading rate image enables and palettes from
    viewports.  The number of enables/palettes now comes from the
    implementation-dependent constant SHADING_RATE_IMAGE_PALETTE_COUNT_NV.  When
    SHADING_RATE_IMAGE_PER_PRIMITIVE_NV (added here) is enabled, the value of
    the new gl_ShadingRateNV built-in output is used to select an enable and
    palette to determine the shading rate.  Otherwise, the viewport number for
    the primitive is used, as in NV_shading_rate_image.


New Procedures and Functions

    None.


New Tokens

    Accepted by the <cap> parameter of Enable, Disable, and IsEnabled and by the
    <pname> parameter of GetBooleanv, GetIntegerv, GetInteger64v, GetFloatv,
    GetDoublev:

        SHADING_RATE_IMAGE_PER_PRIMITIVE_NV             0x95B1

    Accepted by the <pname> parameter of GetBooleanv, GetDoublev,
    GetIntegerv, and GetFloatv:

        SHADING_RATE_IMAGE_PALETTE_COUNT_NV             0x95B2


Modifications to the OpenGL 4.6 Specification (Compatibility Profile)

    Modify Section 14.3.1, Multisampling, as modified by the
    NV_shading_rate_image extension.

    (modify the introduction of the shading rate image functionality to decouple
    shading rate image enables and viewports)

    When using a shading rate image (Section 14.4.1), rasterization may produce
    fragments covering multiple pixels, where each pixel is treated as a sample.
    If any of the SHADING_RATE_IMAGE_NV enables is enabled, primitives will be
    processed with multisample rasterization rules, regardless of the
    MULTISAMPLE enable or the value of SAMPLE_BUFFERS.  ...


    Modify Section 14.4.1, Shading Rate Image, as added by the
    NV_shading_rate_image extension.

    (rework the introduction of the shading rate image functionality to decouple
    shading rate image enables and viewports)

    Applications can specify the use of a shading rate that varies by (x,y)
    location using a _shading rate image_.  For each primitive, the shading rate
    image is enabled or disabled by selecting a single enable in an array of
    enables whose size is given by the implementation-dependent constant
    SHADING_RATE_IMAGE_PALETTE_COUNT_NV.  Use of a shading rate image is enabled
    or disabled globally by using Enable or Disable with target
    SHADING_RATE_IMAGE_NV.  A single shading rate image enable can be modified
    by calling Enablei or Disablei with the constant SHADING_RATE_IMAGE_NV and
    the index of the selected enable.  The shading rate image may only be used
    with a framebuffer object.  When rendering to the default framebuffer, the
    shading rate image enables are ignored and operations in this section are
    disabled.  In the initial state, all shading rate image enables are
    disabled.

    The method used to select a single shading rate image enable used to process
    each primitive is controlled by calling Enable or Disable with the target
    SHADING_RATE_IMAGE_PER_PRIMITIVE_NV.  When enabled, a shading rate enable
    used for a primitive is selected using an index taken from the value of the
    built-in output gl_ShadingRateNV for the primitive's provoking vertex.  If
    the value of gl_ShadingRateNV is negative or greater than or equal to the
    number of shading rate enables, the shading rate used for a primitive is
    undefined.  When SHADING_RATE_IMAGE_PER_PRIMITIVE_NV is disabled, a shading
    rate enable is selected using the index of the viewport used for processing
    the primitive.  In the initial state, SHADING_RATE_IMAGE_PER_PRIMITIVE_NV is
    disabled.


    (rework the introduction of the shading rate image functionality to decouple
    shading rate image palettes and viewports)

    A shading rate index is mapped to a _base shading rate_ using a lookup table
    called the shading rate image palette.  There is a separate palette
    associated with each shading rate image enable.  As with the shading rate
    image enables, a single palette is selected for a primitive according to the
    enable SHADING_RATE_IMAGE_PER_PRIMITIVE_NV.  The number of palettes and the
    number of entries in each palette are given by the implementation-dependent
    constants SHADING_RATE_IMAGE_PALETTE_COUNT_NV and
    SHADING_RATE_IMAGE_PALETTE_SIZE_NV, respectively.  The base shading rate for
    an (x,y) coordinate with a shading rate index of <i> will be given by entry
    <i> of the selected palette.  If the shading rate index is greater than or
    equal to the palette size, the results of the palette lookup are undefined.


    (rework the introduction of ShadingRateImagePaletteNV to decouple shading
    rate image palettes and viewports)

      Shading rate image palettes are updated using the command

        void ShadingRateImagePaletteNV(uint viewport, uint first, sizei count,
                                       const enum *rates);

      <viewport> specifies the number of the palette that should be updated.
      [[ Note:  The formal parameter name <viewport> is a remnant of the
      original NV_shading_rate_image extension, where palettes were tightly
      coupled with viewports. ]]  <rates> is an array ...


    (modify the discussion of ShadingRateImagePaletteNV errors to decouple
    shading rate image palettes and viewports)

        INVALID_VALUE is generated if <viewport> is greater than or equal to
        SHADING_RATE_IMAGE_PALETTE_COUNT_NV.

        INVALID_VALUE is generated if <first> plus <count> is greater than
        SHADING_RATE_IMAGE_PALETTE_SIZE_NV.


    (modify the discussion of GetShadingRateImagePaletteNV to decouple shading
    rate image palettes and viewports)

      Individual entries in the shading rate palette can be queried using the
      command:

        void GetShadingRateImagePaletteNV(uint viewport, uint entry,
                                          enum *rate);

      where <viewport> specifies the number of the palette to query and...
      <entry> specifies the palette entry number.  [[ Note:  The formal
      parameter name <viewport> is a remnant of the original
      NV_shading_rate_image extension, where palettes were tightly coupled
      with viewports. ]]  A single enum from Table X.1 is returned in <rate>.

      Errors

        INVALID_VALUE is generated if <viewport> is greater than or equal to
        SHADING_RATE_IMAGE_PALETTE_COUNT_NV.

        INVALID_VALUE is generated if <first> plus <count> is greater than
        SHADING_RATE_IMAGE_PALETTE_SIZE_NV.


    Modify Section 14.9.3, Multisample Fragment Operations, as edited by
    NV_shading_rate_image.

    (modify the discussion of the shading rate image multisample functionality
    to decouple shading rate image enables and viewports)

    ... This step is skipped if MULTISAMPLE is disabled or if the value of
    SAMPLE_BUFFERS is not one, unless one or more of the SHADING_RATE_IMAGE_NV
    enables are enabled.


Dependencies on NV_mesh_shader

    When NV_mesh_shader is supported, the "NV_primitive_shading_rate" GLSL
    extension allows multi-view mesh shaders to write separate per-primitive
    shading rates for each view using the built-in gl_ShadingRatePerViewNV[].
    When gl_ShadingRatePerViewNV[] is used with
    SHADING_RATE_IMAGE_PER_PRIMITIVE_NV enabled, a separate shading rate image
    enable and palette will be used for each view.


Additions to the AGL/GLX/WGL Specifications

    None

Errors

    See the "Errors" sections for individual commands above.

New State

    Get Value                   Get Command        Type    Initial Value    Description                 Sec.    Attribute
    ---------                   ---------------    ----    -------------    -----------                 ----    ---------
    SHADING_RATE_IMAGE_         IsEnabled           B      FALSE            Use per-primitive shading   14.4.1  enable
      PER_PRIMITIVE_NV                                                      rate to select shading
                                                                            rate images/palettes

New Implementation Dependent State

                                                    Minimum
    Get Value                Type  Get Command      Value   Description                   Sec.
    ---------                ----- ---------------  ------- ------------------------      ------
    SHADING_RATE_IMAGE_      Z+    GetIntegerv      16      Number of shading rate image  14.4.1
      PALETTE_COUNT_NV                                      enables/palettes supported

Issues

    (1) How should we name this extension?

      RESOLVED:  We are calling this extension "NV_primitive_shading_rate"
      because it adds a new per-primitive shading rate to the variable-rate
      shading functionality added by NV_shading_rate_image.

    (2) Do we need to add queries like LAYER_PROVOKING_VERTEX and
        VIEWPORT_INDEX_PROVOKING_VERTEX to determine the vertex used to obtain
        the per-primitive shading rate for each primitive?

      RESOLVED:  No -- we will always use the provoking vertex.  In the event
      that this extension is standardized in the future with behavior that
      diverges between implementations, such a query could be added as part of
      that effort.

Revision History

    Revision 1 (pbrown)
    - Internal revisions.
