# NV_viewport_array

Name

    NV_viewport_array

Name Strings

    GL_NV_viewport_array

Contributors

    Contributors to ARB_viewport_array
    Mathias Heyer, NVIDIA
    James Helferty, NVIDIA
    Daniel Koch, NVIDIA

Contact

    Mathias Heyer, NVIDIA (mheyer 'at' nvidia.com)

Notice

    Copyright (c) 2010-2014 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

    Portions Copyright (c) 2014 NVIDIA Corporation.

Status

    Complete

Version

    Last Modified Date:         10/24/2014
    Author Revision:            5

Number

    OpenGL ES Extension #202

Dependencies

    This extension is written against the OpenGL ES 3.1 (March 14, 2014)
    Specification.

    This extension is written against the OpenGL ES Shading Language
    Specification version 3.10 (March 14, 2014)

    OpenGL ES 3.1 and the EXT_geometry_shader extension are required.

    This extension interacts with EXT_draw_buffers_indexed.

Overview

    OpenGL ES is modeled on a pipeline of operations. The final stage in this
    pipeline before rasterization is the viewport transformation. This stage
    transforms vertices from view space into window coordinates and allows the
    application to specify a rectangular region of screen space into which
    OpenGL should draw primitives. Unextended OpenGL implementations provide a
    single viewport per context. In order to draw primitives into multiple
    viewports, the OpenGL viewport may be changed between several draw calls.
    With the advent of Geometry Shaders, it has become possible for an
    application to amplify geometry and produce multiple output primitives for
    each primitive input to the Geometry Shader. It is possible to direct these
    primitives to render into a selected render target. However, all render
    targets share the same, global OpenGL viewport.

    This extension enhances OpenGL by providing a mechanism to expose multiple
    viewports. Each viewport is specified as a rectangle. The destination
    viewport may be selected per-primitive by the geometry shader. This allows
    the Geometry Shader to produce different versions of primitives destined
    for separate viewport rectangles on the same surface. Additionally, when
    combined with multiple framebuffer attachments, it allows a different
    viewport rectangle to be selected for each. This extension also exposes a
    separate scissor rectangle for each viewport. Finally, the viewport bounds
    are now floating point quantities allowing fractional pixel offsets to be
    applied during the viewport transform.

New Procedures and Functions

    void ViewportArrayvNV(uint first, sizei count, const float * v);
    void ViewportIndexedfNV(uint index, float x, float y, float w, float h);
    void ViewportIndexedfvNV(uint index, const float * v);
    void ScissorArrayvNV(uint first, sizei count, const int * v);
    void ScissorIndexedNV(uint index, int left, int bottom, sizei width, sizei height);
    void ScissorIndexedvNV(uint index, const int * v);
    void DepthRangeArrayfvNV(uint first, sizei count, const float * v);
    void DepthRangeIndexedfNV(uint index, float n, float f);
    void GetFloati_vNV(enum target, uint index, float *data);

    void EnableiNV(enum target, uint index);
    void DisableiNV(enum target, uint index);
    boolean IsEnablediNV(enum target, uint index);

New Tokens

    Accepted by the <pname> parameter of GetBooleanv, GetIntegerv, GetFloatv,
    and GetInteger64v:

        MAX_VIEWPORTS_NV                                0x825B
        VIEWPORT_SUBPIXEL_BITS_NV                       0x825C
        VIEWPORT_BOUNDS_RANGE_NV                        0x825D
        VIEWPORT_INDEX_PROVOKING_VERTEX_NV              0x825F

    Accepted by the <pname> parameter of GetIntegeri_v:

        SCISSOR_BOX                                     0x0C10

    Accepted by the <pname> parameter of GetFloati_vNV:

        VIEWPORT                                        0x0BA2
        DEPTH_RANGE                                     0x0B70

    Accepted by the <pname> parameter of EnableiNV, DisableiNV,
    and IsEnablediNV:

        SCISSOR_TEST                                    0x0C11

Additions to Chapter 11 of the OpenGL ES 3.1 Specification
(Programmable Vertex Processing)

    Modify section Section 11.1gs.4.5 Layer Selection

    Rename the the "Layer Selection" subsection to "Layer and Viewport
    Selection".

    After the first paragraph, insert:

    Geometry shaders may also select the destination viewport for each
    output primitive. The destination viewport for a primitive may be
    selected in the geometry shader by writing to the built-in output
    variable gl_ViewportIndex. This functionality allows a geometry
    shader to direct its output to a different viewport for each
    primitive, or to draw multiple versions of a primitive into several
    different viewports.

    Replace the first two sentences of the second paragraph with:

    The specific vertex of a primitive that is used to select the
    rendering layer or viewport index is implementation-dependent and
    thus portable applications will assign the same layer and viewport
    index for all vertices in a primitive. The vertex conventions
    followed for gl_Layer and gl_ViewportIndex may be determined by
    calling GetIntegerv with the symbolic constants
    LAYER_PROVOKING_VERTEX_EXT and VIEWPORT_INDEX_PROVOKING_VERTEX_NV,
    respectively.

    Modify section 12.5.1 "Controlling the Viewport", page 284.

    Change the first paragraph of section 12.5.1 to read

    The viewport transformation is determined by the selected viewport's
    width and height in pixels, p_x and p_y, respectively, and its
    center (o_x,o_y) (also in pixels) ...

        { leave equations intact }

    Multiple viewports are available and are numbered zero through the
    value of MAX_VIEWPORTS_NV minus one. If a geometry shader is active
    and writes to gl_ViewportIndex, the viewport transformation uses the
    viewport corresponding to the value assigned to gl_ViewportIndex
    taken from an implementation-dependent primitive vertex. If the
    value of the viewport index is outside the range zero to the value
    of MAX_VIEWPORTS_NV minus one, the results of the viewport
    transformation are undefined. If no geometry shader is active, or if
    the active geometry shader does not write to gl_ViewportIndex, the
    viewport numbered zero is used by the viewport transformation.

    A single vertex may be used in more than one individual primitive, in
    primitives such as TRIANGLE_STRIP.  In this case, the viewport
    transformation is applied separately for each primitive.

    The factor and offset applied to Z_d for each viewport encoded by n
    and f are set using

        void DepthRangeArrayfvNV(uint first, sizei count, const float * v);
        void DepthRangeIndexedfNV(uint index, float n, float f);
        void DepthRangef(float n, float f);

    DepthRangeArrayfvNV is used to specify the depth range for multiple
    viewports simultaneously. <first> specifies the index of the first
    viewport to modify and <count> specifies the number of viewports. If
    (<first> + <count>) is greater than the value of MAX_VIEWPORTS_NV then
    an INVALID_VALUE error will be generated. Viewports whose indices
    lie outside the range [<first>, <first> + <count>) are not modified.
    The <v> parameter contains the address of an array of float types
    specifying near (n) and far (f) for each viewport in that order.
    (n) and (f) of each viewport will be clamped to [0.0, 1.0].

    DepthRangeIndexedfNV specifies the depth range for a single viewport
    and is equivalent (assuming no errors are generated) to:

        float v[] = { n, f };
        DepthRangeArrayfvNV(index, 1, v);

    DepthRangef sets the depth range for all viewports to the same values
    and is equivalent (assuming no errors are generated) to:

        for (uint i = 0; i < MAX_VIEWPORTS_NV; i++)
            DepthRangeIndexedfNV(i, n, f);

    Z_w is represented as either ...

    Replace the end of section 12.5.1, starting from "Viewport transformation
    parameters are specified using..."

    Viewport transformation parameters are specified using

        void ViewportArrayvNV(uint first, sizei count, const float * v);
        void Viewport(int x, int y, sizei w, sizei h);
        void ViewportIndexedfNV(uint index, float x, float y, float w, float h);
        void ViewportIndexedfvNV(uint index, const float * v);

    ViewportArrayvNV specifies parameters for multiple viewports
    simultaneously. <first> specifies the index of the first viewport to
    modify and <count> specifies the number of viewports. If (<first> +
    <count>) is greater than the value of MAX_VIEWPORTS_NV then an
    INVALID_VALUE error will be generated. Viewports whose indices lie
    outside the range [<first>, <first> + <count>) are not modified.
    <v> contains the address of an array of floating point values
    specifying the left (x), bottom (y), width (w) and height (h) of
    each viewport, in that order. <x> and <y> give the location of the
    viewport's lower left corner and <w> and <h> give the viewport's
    width and height, respectively.

    ViewportIndexedfNV and ViewportIndexedfvNV specify parameters for a
    single viewport and are equivalent (assuming no errors are
    generated) to:

        float v[4] = { x, y, w, h };
        ViewportArrayvNV(index, 1, v);

    and

        ViewportArrayvNV(index, 1, v);

    respectively.

    Viewport sets the parameters for all viewports to the same values
    and is equivalent (assuming no errors are generated) to:

        for (uint i = 0; i < MAX_VIEWPORTS_NV; i++)
            ViewportIndexedfNV(i, (float)x, (float)y, (float)w, (float)h);

    The viewport parameters shown in the above equations are found from these
    values as

        o_x = x + w /2,
        o_y = y + h / 2,
        p_x = w,
        p_y = h.

    The location of the viewport's bottom-left corner, given by (x,y), are
    clamped to be within the implementation-dependent viewport bounds range.
    The viewport bounds range [min, max] tuple may be determined by
    calling GetFloatv with the symbolic constant VIEWPORT_BOUNDS_RANGE_NV
    (see chapter 20).

    Viewport width and height are clamped to implementation-dependent maximums
    when specified. The maximum width and height may be found by calling
    GetFloatv with the symbolic constant MAX_VIEWPORT_DIMS. The maximum
    viewport dimensions must be greater than or equal to the larger of
    the visible dimensions of the display being rendered to (if a
    display exists), and the largest renderbuffer image which can be
    successfully created and attached to a framebuffer object (see
    chapter 9). INVALID_VALUE is generated if either w or h is negative.

    The state required to implement the viewport transformations is four
    floating-point values and two clamped floating-point values for each
    viewport. In the initial state, w and h for each viewport are set to
    the width and height, respectively, of the window into which the GL
    is to do its rendering. If the default framebuffer is bound but no
    default framebuffer is associated with the GL context (see chapter
    9), then w and h are initially set to zero. o_x and o_y are set to
    w/2 and h/2, respectively. n and f are set to 0.0 and 1.0,
    respectively.

    The precision with which the GL interprets the floating point viewport
    bounds is implementation-dependent and may be determined by querying the
    implementation-defined constant VIEWPORT_SUBPIXEL_BITS_NV.

Additions to Chapter 15 of the OpenGL ES 3.1 Specification (Writing
Fragments and Samples to the Framebuffer)

    Replace section 15.1.2 "Scissor Test", page 309.

    The scissor test determines if (xw, yw) lies within the scissor rectangle
    defined by four values for each viewport. These values are set with

        void ScissorArrayvNV(uint first, sizei count, const int * v);
        void ScissorIndexedNV(uint index, int left, int bottom, sizei width, sizei height);
        void ScissorIndexedvNV(uint index, int * v);
        void Scissor(int left, int bottom, sizei width, sizei height);

    ScissorArrayvNV defines a set of scissor rectangles that are each
    applied to the corresponding viewport (see section 12.5.1
    "Controlling the Viewport"). <first> specifies the index of the
    first scissor rectangle to modify, and <count> specifies the number
    of scissor rectangles. If (<first> + <count>) is greater than the
    value of MAX_VIEWPORTS_NV, then an INVALID_VALUE error is generated.
    <v> contains the address of an array of integers containing the
    left, bottom, width and height of the scissor rectangles, in that
    order.

    If left <= x_w < left + width and bottom <= y_w < bottom + height
    for the selected scissor rectangle, then the scissor test passes.
    Otherwise, the test fails and the fragment is discarded. For points,
    lines, and polygons, the scissor rectangle for a primitive is
    selected in the same manner as the viewport (see section 12.5.1).

    The scissor test is enabled or disabled for all viewports using
    Enable or Disable with the symbolic constant SCISSOR_TEST. The test
    is enabled or disabled for a specific viewport using EnableiNV or
    DisableiNV with the constant SCISSOR_TEST and the index of the
    selected viewport. When disabled, it is as if the scissor test
    always passes. The value of the scissor test enable for viewport <i>
    can be queried by calling IsEnablediNV with <target> SCISSOR_TEST and
    <index> <i>. The value of the scissor test enable for viewport zero
    may also be queried by calling IsEnabled with the same symbolic
    constant, but no <index> parameter. If either width or height is
    less than zero for any scissor rectangle, then an INVALID_VALUE
    error is generated. If the viewport index specified to EnableiNV,
    DisableiNV or IsEnablediNV is greater or equal to the value of
    MAX_VIEWPORTS_NV, then an INVALID_VALUE error is generated.

    The state required consists of four integer values per viewport, and
    a bit indicating whether the test is enabled or disabled for each
    viewport. In the initial state, left = bottom = 0, and width and
    height are determined by the size of the window into which the GL is
    to do its rendering for all viewports. If the default framebuffer is
    bound but no default framebuffer is associated with the GL context
    (see chapter 9), then with and height are initially set to zero.
    Initially, the scissor test is disabled for all viewports.

    ScissorIndexedNV and ScissorIndexedvNV specify the scissor rectangle for
    a single viewport and are equivalent (assuming no errors are
    generated) to:

        int v[] = { left, bottom, width, height };
        ScissorArrayvNV(index, 1, v);

    and

        ScissorArrayvNV(index, 1, v);

    respectively.

    Scissor sets the scissor rectangle for all viewports to the same
    values and is equivalent (assuming no errors are generated) to:

        for (uint i = 0; i < MAX_VIEWPORTS_NV; i++) {
            ScissorIndexedNV(i, left, bottom, width, height);
        }

    Calling Enable or Disable with the symbolic constant SCISSOR_TEST is
    equivalent, assuming no errors, to:

    for (uint i = 0; i < MAX_VIEWPORTS_NV; i++) {
        EnableiNV(SCISSOR_TEST, i);
        /* or */
        DisableiNV(SCISSOR_TEST, i);
    }

Additions to Chapter 19 of the OpenGL ES 3.1 Specification (Context State
Queries)

    Modifications to Section 19.1 Simple Queries

        Add to the list of indexed query functions:

        void GetFloati_vNV(enum target, uint index, float *data);

Additions to the OpenGL ES Shading Language Version 3.10 Specification

    Add a new Section 3.4.x, GL_NV_viewport_array Extension (p. 13)

    3.4.x GL_NV_viewport_array Extension

    To use the GL_NV_viewport_array extension in a shader it must be
    enabled using the #extension directive.

    The shading language preprocessor #define GL_NV_viewport_array will
    be defined to 1 if the GL_NV_viewport_array extension is supported.

    Modify Section 7.1.1gs, "Geometry Shader Special Variables"

    Add to the list of geometry shader built-in variables:

        out highp int gl_ViewportIndex;    // may be written to


    Additions to Section 7.1.1gs.2, "Geometry Shader Output Variables"

    Add a paragraph after the paragraph describing gl_Layer, starting
    "gl_Layer is used to select a specific layer (or face and layer of a
    cube map) of a multi-layer framebuffer attachment.":

    The built-in variable gl_ViewportIndex is available as an output variable
    in the geometry shader and an input variable in the fragment shader. In the
    geometry shader it provides the index of the viewport to which the next
    primitive emitted from the geometry shader should be drawn. Primitives
    generated by the geometry shader will undergo viewport transformation and
    scissor testing using the viewport transformation and scissor rectangle
    selected by the value of gl_ViewportIndex. The viewport index used will
    come from one of the vertices in the primitive being shaded. Which vertex
    the viewport index comes from is implementation-dependent, so it is best to
    use the same viewport index for all vertices of the primitive. If a
    geometry shader does not assign a value to gl_ViewportIndex, viewport
    transform and scissor rectangle zero will be used. If a geometry shader
    assigns a value to gl_ViewportIndex and there is a path through the shader
    that does not set gl_ViewportIndex, then the value of gl_ViewportIndex is
    undefined for executions of the shader that take that path. See section
    11.1gs.4 "Geometry Shader Outputs" of the OpenGL ES Specification for more
    information.

    Modify section 7.1.2 "Fragment Shader Special Variables", as modified by
    EXT_geometry_shader:

    Add to the list of built-in variables:

        in highp int gl_ViewportIndex;

    Add description of the variable:

    The input variable gl_ViewportIndex will have the same value that was
    written to the output variable gl_ViewportIndex in the geometry stage. If
    the geometry stage does not dynamically assign to gl_ViewportIndex, the
    value of gl_ViewportIndex in the fragment shader will be undefined. If the
    geometry stage makes no static assignment to gl_ViewportIndex, the fragment
    stage will read zero. Otherwise, the fragment stage will read the same
    value written by the geometry stage, even if that value is out of range. If
    a fragment shader contains a static access to gl_ViewportIndex, it will
    count against the implementation defined limit for the maximum number of
    inputs to the fragment stage.

    Add to Section 7.2 "Built-In Constants", as modified by
    EXT_geometry_shader, to the list of built-in constants matching the
    corresponding API implementation-dependent limits:

        const highp int gl_MaxViewports = 16;

Errors

    INVALID_VALUE is generated by ViewportArrayvNV if <first> + <count> is
    greater than or equal to the value of MAX_VIEWPORTS_NV, or if any
    viewport's width or height is less than 0.

    INVALID_VALUE is generated by ScissorArrayvNV if <first> + <count> is
    greater than or equal to the value of MAX_VIEWPORTS_NV, or if any
    scissor rectangle's width or height is less than zero.

    INVALID_VALUE is generated by DepthRangeArrayfvNV if <first> + <count> is
    greater than or equal to the vaue of MAX_VIEWPORTS_NV.

    INVALID_VALUE is generated by EnableiNV, DisableiNV and IsEnablediNV if
    <index> is greater than or equal to the value of MAX_VIEWPORTS_NV.

New State

    Table 20.5 (p. 356)

    Get Value                 Type             Get Command       Initial Value   Description                 Sec
    ------------------------  ---------------- ------------      -------------   --------------------------  -----
    VIEWPORT                  16* x 4 x R      GetFloati_vNV       See 2.11.1      Viewport origin & extent  12.5.1
    DEPTH_RANGE               16* x 2 x R[0,1] GetFloati_vNV       See 2.16.1      Depth range near & far    12.5.1

NOTE: The changes are that VIEWPORT and DEPTH_RANGE are extended to
accommodate 16* copies and now consist of floating-point and
double-precision values, respectively.

    Table 20.12 (p. 363)

    Get Value                 Type        Get Command           Initial Value   Description               Sec
    ------------------------  ----------  -------------         -------------   -------------------       ------
    SCISSOR_TEST              16* x B     IsEnablediNV          FALSE           Scissoring enabled        15.1.2
    SCISSOR_BOX               16* x 4 x Z GetIntegeri_v         See 4.1.2       Scissor box               15.1.2

NOTE: The only change is that SCISSOR_TEST and SCISSOR_BOX are extended
to accommodate 16* copies.

New Implementation Dependent State

    Get Value                          Type   Get Command     Minimum Value   Description                     Sec.
    ---------                          ----   -----------     -------------   -------------------             -----
    MAX_VIEWPORT_DIMS    (NOTE 1)      2 x Z+ GetFloatv       See 2.16.1      Maximum viewport dimensions     12.5.1
    MAX_VIEWPORTS_NV                   Z+     GetIntegerv     16              Maximum number of               12.5.1
                                                                              active viewports
    VIEWPORT_SUBPIXEL_BITS_NV          Z+     GetIntegerv     0               Number of bits of sub-pixel     12.5.1
                                                                              precision for viewport bounds
    VIEWPORT_BOUNDS_RANGE_NV           2 x R  GetFloatv       (NOTE 2)        Viewport bounds range [min,max] 12.5.1
    LAYER_PROVOKING_VERTEX_NV          Z_4    GetIntegerv     -- (NOTE 3)     vertex convention followed by   12.5.1
                                                                              the gl_Layer GLSL variable
    VIEWPORT_INDEX_PROVOKING_VERTEX_NV Z_4    GetIntegerv     -- (NOTE 3)     vertex convention followed by   12.5.1
                                                                              the gl_ViewportIndex GLSL
                                                                              variable

NOTE 1: The recommended get command is changed from GetIntegerv to GetFloatv.
NOTE 2: range for viewport bounds:
  * On ES3.1-capable hardware the VIEWPORT_BOUNDS_RANGE_NV should be at least
    [-32768, 32767].
NOTE 3: Valid values are: FIRST_VERTEX_CONVENTION_NV,
LAST_VERTEX_CONVENTION_NV, UNDEFINED_VERTEX_NV.


Interactions with EXT_draw_buffers_indexed

    If EXT_draw_buffers_indexed is supported, EnableiNV, DisableiNV and
    IsEnablediNV alias EnableiEXT, DisableiEXT and IsEnablediEXT, respectively.


Issues

    See issues section in ARB_viewport_array.

    #1 What are the differences from ARB_viewport_array?

    - OpenGL ES does not support the double datatype. The changed interfaces of
    glDepthRangeArrayfvNV and DepthRangeIndexedfNV reflect that. 'float' is
    being used instead of 'clampf', with additional constraints in the text
    that the values will get clamped.
    - The ability to access gl_ViewportIndex from the fragment shader was added
    from ARB_fragment_layer_viewport.


Revision History

    Rev.    Date      Author    Changes
    ----  --------    --------  -----------------------------------------
     1    06/18/2014  mheyer    Based on ARB_viewport_array, stripped for ES3.1
                                - replaced clampd with float for glDepthRangef
                                - instead of EnableIndexed and DisableIndexed, use
                                  Enablei and Disablei
                                - PROVOKING_VERTEX_NV removed
     2    07/24/2014  mheyer    Minor edits.
     3    08/10/2014  mheyer    Edit for consistency.
     4    09/04/2014  jhelferty Add viewport part of ARB_fragment_layer_viewport
                                as was done with layer in EXT_geometry_shader
     5    10/24/2014  dkoch     Cleanup for publishing.
