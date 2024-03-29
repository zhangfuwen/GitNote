# NV_stereo_view_rendering

Name

    NV_stereo_view_rendering

Name Strings

    GL_NV_stereo_view_rendering

Contact

    Kedarnath Thangudu, NVIDIA Corporation (kthangudu 'at' nvidia.com)

Contributors

    Mark Kilgard, NVIDIA Corporation
    Pat Brown, NVIDIA Corporation
    Jason Schmidt, NVIDIA Corporation

Status

    Shipping in NVIDIA release 367.XX drivers and up.

Version

    Last Modified Date:         November 25, 2017
    NVIDIA Revision:            3

Number

    OpenGL Extension #489
    OpenGL ES Extension #296

Dependencies

    This extension is written against OpenGL 4.5 Specification
    (Compatibility Profile).

    This extension interacts with the OpenGL ES 3.1 Specification.

    The extension requires NV_viewport_array2.

    This extension interacts with NV_gpu_program4 and NV_gpu_program5.

    This extension interacts with EXT_tessellation_shader.

Overview

    Virtual reality (VR) applications often render a single logical scene
    from multiple views corresponding to a pair of eyes.  The views (eyes) are
    separated by a fixed offset in the X direction.

    Traditionally, multiple views are rendered via multiple rendering passes.
    This is expensive for the GPU because the objects in the scene must be
    transformed, rasterized, shaded, and fragment processed redundantly.  This
    is expensive for the CPU because the scene graph needs to be visited
    multiple times and driver validation happens for each view.  Rendering N
    passes tends to take N times longer than a single pass.

    This extension provides a mechanism to render binocular (stereo) views
    from a single stream of OpenGL rendering commands.  Vertex, tessellation,
    and geometry (VTG) shaders can output two positions for each vertex
    corresponding to the two eye views.  A built-in "gl_SecondaryPositionNV"
    is added to specify the second position.  The positions from each view may
    be sent to different viewports and/or layers.  A built-in
    "gl_SecondaryViewportMaskNV[]" is also added to specify the viewport mask
    for the second view.  A new layout-qualifier "secondary_view_offset" is
    added for built-in output "gl_Layer" which allows for the geometry from
    each view to be sent to different layers for rendering.

New Procedures and Functions

    None.

New Tokens

    None.

Additions to Chapter 11 of the OpenGL 4.5 (Compatibility Profile) Specification
(Programmable Vertex Processing)

    Add to Section 11.1.3.10 (Shader Outputs)

    The built-in output variables gl_SecondaryPositionNV and
    gl_SecondaryViewportMaskNV[] hold the homogeneous vertex position and the
    viewport mask for the secondary view, as described in Section 11.5
    (Stereo Offsetted Rendering).

    [Section 11.2 of the OpenGL 4.5 specification corresponds to
     Section 11.1ts of the OpenGL ES 3.1 specification, and
     Section 11.3 of the OpenGL 4.5 specification corresponds to
     Section 11.1gs of the OpenGL ES 3.1 specification]

    In Section 11.2.1.2.2 (Tessellation Control Shader Inputs), modify the
    line describing members of gl_in[] to include gl_SecondaryPositionNV

    The members of each element of the gl_in[] array are gl_Position,
    gl_SecondaryPositionNV, gl_PointSize, ...

    Add to Section 11.2.1.2.3 (Tessellation Control Shader Outputs)

    The built-in output variables gl_SecondaryPositionNV and
    gl_SecondaryViewportMaskNV[] hold the homogeneous vertex position and the
    viewport mask for the secondary view, as described in Section 11.5
    (Stereo Offsetted Rendering).

    In Section 11.2.3.3 (Tessellation Evaluation Shader Outputs), modify the
    line describing members of gl_in[] to include gl_SecondaryPositionNV

    The members of each element of the gl_in[] array are gl_Position,
    gl_SecondaryPositionNV, gl_PointSize, ...

    Add to Section 11.2.3.4 (Tessellation Evaluation Shader Outputs)

    The built-in output variables gl_SecondaryPositionNV and
    gl_SecondaryViewportMaskNV[] hold the homogeneous vertex position and the
    viewport mask for the secondary view, as described in Section 11.5
    (Stereo Offsetted Rendering).

    In Section 11.3.4.4 (Geometry Shader Inputs), add to the list of members
    of gl_in[]

    The members of each element of the gl_in[] array are:
        ...
        * Structure member gl_SecondaryPositionNV holds the per-vertex position
        of the secondary view, written by the upstream shader to the built-in
        output variable gl_SecondaryPositionNV. Note that writing to
        gl_SecondaryPositionNV from either the upstream or geometry shader is
        optional (also see section 7.1("Built-In Variables") of the OpenGL
        Shading Language Specification).

    Add to Section 11.2.3.5 (Geometry Shader Outputs)

    The built-in output variables gl_SecondaryPositionNV and
    gl_SecondaryViewportMaskNV[] hold the homogeneous vertex position and the
    viewport mask for the secondary view, as described in Section 11.5
    (Stereo Offsetted Rendering).


    Modify Section 11.4 (Layer and Viewport Selection) to include
    gl_SecondaryViewportMaskNV[] wherever gl_ViewportMask[] is mentioned.

    Add a new Section 11.5 (Stereo Offsetted Rendering)

    Application may render a single logical scene from two views,
    corresponding to a pair of eyes, that are separated only in the X
    direction by fixed offset. Vertex, tessellation, and geometry (VTG)
    shaders may compute and write the vertex coordinates for each of the pair
    of stereo views.  The input/output built-ins "gl_Position" and
    "gl_SecondaryPositionNV" may be used to read/write the positions
    corresponding to the primary and secondary views respectively.  Only the
    "x" coordinate is expected to differ between these views, so the "yzw"
    coordinates for secondary position are also obtained from the "gl_Position"
    and writes to "gl_SecondaryPositionNV.yzw" are ignored.

    VTG shaders may also write to "gl_SecondaryViewportMaskNV[]" to specify a
    viewport mask for primitives in the secondary view. If
    "gl_SecondaryViewportMaskNV[]" is not specified, "gl_ViewportMask[]" will
    be used as the viewport mask for primitives from both views.

    Primitives from each view may be sent to different layers as described
    in Section 7.1 of the GLSL specification.

Additions to Chapter 13 of the OpenGL 4.5 (Compatibility Profile)
Specification (Fixed-Function Vertex Post-Processing)

    Modify Section 13.2 (Transform Feedback), p. 453 [section 12.1 in OpenGL ES]

    Modify the first paragraph:

    ...The vertices are fed back after vertex color clamping, but before
    viewport mask expansion, stereo view expansion, flat-shading, and
    clipping. ...

    Modify Section 13.6.1 (Controlling the Viewport)
    [section 12.5.1 in OpenGL ES] to include
    gl_SecondaryViewportMaskNV[] wherever gl_ViewportMask[] is mentioned.


New Implementation Dependent State

    None.

New State

    None.

Additions to the AGL/GLX/WGL/EGL Specifications

    None.

GLX Protocol

    None.

Modifications to the OpenGL Shading Language Specification, Version 4.50

    Including the following line in a shader can be used to control the
    language features described in this extension:

        #extension GL_NV_stereo_view_rendering : <behavior>

    where <behavior> is as specified in section 3.3

    New preprocessor #defines are added to the OpenGL Shading Language:

        #define GL_NV_stereo_view_rendering     1

    Modify Section 7.1 (Built-In Language Variables), p. 118

    Add to the list of vertex shader built-ins:

        out gl_PerVertex {
            highp vec4 gl_SecondaryPositionNV;
            highp int  gl_SecondaryViewportMaskNV[];
        };

    Add to the list of geometry shader built-ins:

        in gl_PerVertex {
            highp vec4 gl_SecondaryPositionNV;
        } gl_in[];

        out gl_PerVertex {
            highp vec4 gl_SecondaryPositionNV;
            highp int gl_SecondaryViewportMaskNV[];
        };

    Add to the list of tessellation control shader built-ins:

        in gl_PerVertex {
            highp vec4 gl_SecondaryPositionNV;
        } gl_in[gl_MaxPatchVertices];

        out gl_PerVertex {
            highp vec4 gl_SecondaryPositionNV;
            highp int  gl_SecondaryViewportMaskNV[];
        } gl_out[];

    Add to the list of tessellation evaluation shader built-ins:

        in gl_PerVertex {
            highp vec4 gl_SecondaryPositionNV;
        } gl_in[gl_MaxPatchVertices];

        out gl_PerVertex {
            highp vec4 gl_SecondaryPositionNV;
            highp int  gl_SecondaryViewportMaskNV[];
        };

    Add the following descriptions for gl_SecondaryPositionNV and
    gl_SecondaryViewportMaskNV[]:

    The output variables gl_SecondaryPositionNV and gl_SecondaryViewportMaskNV[]
    are available in vertex, tessellation, and geometry languages to specify
    the position and the viewport mask for the secondary view respectively.
    The input variable gl_SecondaryPositionNV is available in the tessellation
    and geometry languages to read the secondary position written by a
    previous shader stage. See section 11.5 ("Stereo Offsetted Rendering") of
    the OpenGL 4.5 specification for more information.


    Add the following to the description of gl_Layer

    The shader output gl_Layer may be redeclared with the following layout
    qualifier which is available in vertex, tessellation, and geometry
    shaders:

        layout-qualifier-id
            secondary_view_offset = integer-constant-expression

    along with the <viewport-relative> layout qualifier introduced in
    NV_viewport_array2.  When using layered rendering, the layout qualifier
    <secondary_view_offset> may be used to indicate the layer offset for the
    primitives in the second view. If gl_Layer is redeclared with both
    <viewport_relative> and <secondary_view_offset>, the layer used for
    rendering the primitives of the second view is computed by first adding
    the viewport index and then the offset value specified by
    <secondary_view_offset>.


Errors

    None.

Interactions with EXT_tessellation_shader

    If implemented on OpenGL ES and EXT_tessellation_shader is not supported,
    remove all language referring to the tessellation control and tessellation
    evaluation pipeline stages.

Interactions with NV_gpu_program4 and NV_gpu_program5

    If NV_gpu_program4 is supported and the "NV_stereo_view_rendering" program
    option is specified, vertex, tessellation control/evaluation, and geometry
    program result variable "result.secondaryposition" can be used to specify
    the vertex's position coordinates from the second view and
    "result.secondaryviewportmask[]" can be used to specify the mask of
    viewports that the primitives from the second view will be emitted to. When
    this program option is specified in tessellation control/evaluation, and/or
    geometry programs, a vertex attribute "vertex[m].secondaryposition" is also
    available to read the secondary position computed in a previous shader
    stage.

    Modify Section 2.X.2 of NV_gpu_program4, Program Grammar

    (add the following rule to the NV_gpu_program4 grammar for Geometry and
    Tessellation Control/Eval programs)

    <attribBasic>       ::= ...
                          | <vtxPrefix> "secondaryposition"

    (add the following rule to the NV_gpu_program4 grammar for Vertex,
    Geometry, and Tessellation Control/Eval programs)

    <declaration>       ::= ...
                          | "SECONDARY_VIEW_LAYER_OFFSET" <optSign> <int>

    <resultBasic>       ::= ...
                          | <resPrefix> "secondaryposition"
                          | <resPrefix> "secondaryviewportmask" arrayMemAbs

    (add the following to the tables of Geometry and Tessellation
    Control/Eval Program Attribute Bindings)

        Binding                         Components  Description
        ---------------------------     ----------  ---------------------------
        vertex[m].secondaryposition      (x,y,z,w)  object coordinates for the
                                                    second view

    (add the following subsection to section 2.X.3.2 of NV_gpu_program4,
     Program Attribute Variables)

    If an attribute binding in a geometry/tessellation program matches
    "vertex[m].secondaryposition", the "x" component of the attribute provides
    the "x" coordinate for the secondary view position. The "y", "z", and "w"
    components will be the same as the "y", "z", and "w" components of
    "vertex[m].position".


    (add the following to the tables of Vertex, Geometry, and Tessellation
    Control/Eval Program Result Variable Bindings)

        Binding                        Components  Description
        -----------------------------  ----------  ----------------------------
        result.secondaryposition        (x,y,z,w)  object coordinates for the
                                                   second view
        result.secondaryviewportmask[]  (v,-,-,-)  viewport array mask for the
                                                   second view

    (add the following subsection to section 2.X.3.5 of NV_gpu_program4,
     Program Results.)

    If a result variable binding matches "result.secondaryposition", updates
    to the "x" component of the result variable provide the "x" coordinate for
    the position from the secondary view. The y, z, and w coordinates for the
    secondary view position are expected to be the same as the primary
    position and are taken from the "result.position". Updates to y, z, and w
    components of "result.secondaryposition" are ignored.

    If a result variable binding matches "result.secondaryviewportmask[]",
    updates to the "x" component of the result variable provide a single
    integer that serves as a mask of viewport indices for the secondary
    view. The mask must be written as an integer value; writing a floating-
    point value will produce undefined results. If the value has bits greater
    than or equal to MAX_VIEWPORTS set, the number of viewports the primitive
    is emitted to and which viewports are used undefined. If a program
    specifies "NV_stereo_view_rendering" program option and does not write the
    "result.secondaryviewportmask[]", then "result.viewportmask[]" will be
    used as the viewport mask for both the views.

    If the "NV_stereo_view_rendering" program option is not specified, the
    "result.secondaryposition" and "result.secondaryviewportmask[]" bindings
    are unavailable.

    (add the following to Section 2.X.6.Y, Program Options)

    + Stereo Offsetted Rendering (NV_stereo_view_rendering)

    If a vertex, geometry, tessellation control, or tessellation evaluation
    program specifies the "NV_stereo_view_rendering" option, the result
    bindings "result.secondaryposition" and "result.secondaryviewportmask[]"
    will be available to specify the secondary position and viewportmask
    respectively.

    (add the following to Section 2.X.7.Y, Program Declarations)

    + Layer Offset for the Secondary View (NV_SECONDARY_VIEW_LAYER_OFFSET)

    The NV_SECONDARY_VIEW_LAYER_OFFSET statement declares the layer offset
    value to be added to the "result.layer" for the secondary view. If the
    program also specifies the "NV_layer_viewport_relative" option, both the
    viewport index and the above offset are added to the "result.layer" for
    the second view.

Issues

    (1) Where does the stereo view expansion occur?

    RESOLVED: This operation occurs right before viewport mask expansion
    (NV_viewport_array2). The primary primitive is broadcast to different
    viewports based on the viewport id/viewport mask followed by the secondary
    primitive. This specification applies the stereo view expansion after
    transform feedback, and makes all primitive queries except for clipping
    primitives only count each primitive once. Clipping primitives' queries
    count the primitives once for each view.

    (2) Do we need "gl_SecondaryViewportIdNV" for the second view?

    RESOLVED: No. "gl_SecondaryViewportMaskNV[]" should cover that functionality.

    (3) Should the secondary position be readable from the tessellation
    control/evaluation and geometry shaders?

    RESOLVED: Yes. If the secondary position is written by the vertex shader,
    tessellation shader should be able to read position to evaluate the
    interpolated positions for the second view.

Revision History

    Revision 3 2017/11/25 (pbrown)
      - Add to the OpenGL ES Extension Registry
    Revision 2 2017/02/21 (jaschmidt)
      - Formally add OpenGL ES interactions
    Revision 1
      - Internal revisions.
