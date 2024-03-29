# NV_viewport_array2

Name

    NV_viewport_array2

Name Strings

    GL_NV_viewport_array2

Contact

    Jeff Bolz, NVIDIA Corporation (jbolz 'at' nvidia.com)
    Pat Brown, NVIDIA Corporation (pbrown 'at' nvidia.com)

Contributors

    Pat Brown, NVIDIA
    Mathias Heyer, NVIDIA

Status

    Shipping

Version

    Last Modified Date:         March 27, 2015
    Revision:                   2

Number

    OpenGL Extension #476
    OpenGL ES Extension #237

Dependencies

    This extension is written against the OpenGL 4.3 specification
    (Compatibility Profile).

    This extension interacts with the OpenGL ES 3.1 (March 17, 2014)
    specification.

    If implemented in OpenGL ES, NV_viewport_array, EXT_geometry_shader
    and EXT_shader_io_blocks are required.

    This extension interacts with EXT_tessellation_shader.

    This extension interacts with NV_geometry_shader_passthrough.

    This extension interacts with NV_gpu_program4.

Overview

    This extension provides new support allowing a single primitive to be
    broadcast to multiple viewports and/or multiple layers.  A shader output
    gl_ViewportMask[] is provided, allowing a single primitive to be output to
    multiple viewports simultaneously.  Also, a new shader option is provided
    to control whether the effective viewport index is added into gl_Layer.
    These capabilities allow a single primitive to be output to multiple
    layers simultaneously.

    The gl_ViewportMask[] output is available in vertex, tessellation
    control, tessellation evaluation, and geometry shaders. gl_ViewportIndex
    and gl_Layer are also made available in all these shader stages. The
    actual viewport index or mask and render target layer values are taken
    from the last active shader stage from this set of stages.

    This extension is a superset of the GL_AMD_vertex_shader_layer and
    GL_AMD_vertex_shader_viewport_index extensions, and thus those extension
    strings are expected to be exported if GL_NV_viewport_array2 is
    supported. This extension includes the edits for those extensions, recast
    against the reorganized OpenGL 4.3 specification.

New Procedures and Functions

    None.

New Tokens

    None.

Additions to Chapter 11 of the OpenGL 4.3 (Compatibility Profile) Specification
(Programmable Vertex Processing)

    Add to Section 11.1.3.10 (Shader Outputs)

    The built-in output variables gl_ViewportIndex, gl_ViewportMask[], and 
    gl_Layer hold the viewport index/mask and render target layer, as described
    in Section 11.4 (Viewport Index and Layer).

    Add to Section 11.2.1.2.3 (Tessellation Control Shader Outputs)

    The built-in output variables gl_ViewportIndex, gl_ViewportMask[], and 
    gl_Layer hold the viewport index/mask and render target layer, as described
    in Section 11.4 (Viewport Index and Layer).

    Add to Section 11.2.3.4 (Tessellation Evaluation Shader Outputs)

    The built-in output variables gl_ViewportIndex, gl_ViewportMask[], and 
    gl_Layer hold the viewport index/mask and render target layer, as described
    in Section 11.4 (Viewport Index and Layer).

    Modify Section 11.3.4.5 (Geometry Shader Outputs)

    Replace the paragraph about gl_ViewportIndex:

    The built-in output variables gl_ViewportIndex, gl_ViewportMask[], and 
    gl_Layer hold the viewport index/mask and render target layer, as described
    in Section 11.4 (Viewport Index and Layer).

    Replace Section 11.3.4.6 (Layer and Viewport Selection) with new Section
    11.4 (Layer and Viewport Selection)

    Geometry may be rendered to one of several different layers of cube map
    textures, three-dimensional textures, or one- or two-dimensional texture
    arrays. This functionality allows an application to bind an entire complex
    texture to a framebuffer object, and render primitives to arbitrary layers
    computed at run time. For example, it can be used render a scene into
    multiple layers of an array texture in one pass, or to select a particular
    layer to render to in shader code. The layer to render to is specified by
    writing to the built-in output variable gl_Layer.  Layered rendering
    requires the use of framebuffer objects (see section 9.8).
    
    Shaders may also direct each primitive to zero or more viewports. The 
    destination viewports for a primitive may be selected in the shader by 
    writing to the built-in output variable gl_ViewportIndex (selecting a 
    single viewport) or gl_ViewportMask[] (selecting multiple viewports). 
    This functionality allows a shader to direct its output to different 
    viewports for each primitive, or to draw multiple versions of a primitive 
    into several different viewports.

    The specific vertex of a primitive used to select the rendering layer or
    viewport index/mask is implementation-dependent and thus portable
    applications will assign the same layer and viewport index for all
    vertices in a primitive. The vertex conventions followed for gl_Layer and
    gl_ViewportIndex/gl_ViewportMask[] may be determined by calling
    GetIntegerv with the symbolic constants LAYER_PROVOKING_VERTEX and
    VIEWPORT_INDEX_PROVOKING_VERTEX, respectively. For either query, if the
    value returned is PROVOKING_VERTEX, then vertex selection follows the
    convention specified by ProvokingVertex (see section 13.4). If the value
    returned is FIRST_VERTEX_CONVENTION, selection is always taken from the
    first vertex of a primitive. If the value returned is
    LAST_VERTEX_CONVENTION, the selection is always taken from the last vertex
    of a primitive. If the value returned is UNDEFINED_VERTEX, the selection
    is not guaranteed to be taken from any specific vertex in the
    primitive. The vertex considered the provoking vertex for particular
    primitive types is given in table 13.2.

    The layer selection may be made a function of the viewport index, as 
    described in Section 7.1 of the GLSL specification.

    The viewport index, viewport mask, and layer outputs are available in 
    vertex, tessellation control, tessellation evaluation, and geometry 
    shaders. Only the last active shader stage (in pipeline order) from this
    list controls the viewport index/mask and layer; outputs in previous 
    shader stages are not used, even if the last stage fails to write one of 
    the outputs.


Additions to Chapter 13 of the OpenGL 4.3 (Compatibility Profile) Specification
(Fixed-Function Vertex Post-Processing)

    Modify Section 13.2 (Transform Feedback), p. 453

    Modify the first paragraph:

    ...The vertices are fed back after vertex color clamping, but before
    viewport mask expansion, flatshading, and clipping. ...


    Modify Section 13.6.1 (Controlling the Viewport)
    
    Multiple viewports are available and are numbered zero through the value
    of MAX_VIEWPORTS minus one.  If last active vertex, tessellation, or
    geometry shader writes to gl_ViewportIndex, the primitive is emitted to
    the viewport corresponding to the value assigned to gl_ViewportIndex, as
    taken from an implementation-dependent provoking vertex.  The primitive is
    then transformed using the state of the selected viewport.  If the value
    of the viewport index is outside the range zero to the value of
    MAX_VIEWPORTS minus one, the results of the viewport transformation are
    undefined.
    
    If last active vertex, tessellation, or geometry shader writes to
    gl_ViewportMask[], the primitive is emitted to zero or more viewports.  If
    bit <i> is set in the mask, the primitive is emitted to viewport <i> and
    transformed using the state of viewport <i>.  However, each primitive will
    still be captured by transform feedback and counted by primitive queries
    only once.  If bits of gl_ViewportMask[] greater than or equal to the
    value of MAX_VIEWPORTS are set, the number of times the primitive is
    emitted and which viewport transformations are used are undefined.

    If neither gl_ViewportIndex nor gl_ViewportMask[] are written, the
    viewport numbered zero is used by the viewport transformation.


    Modify Section 14.5.2.1 (Line Stipple)

    (add to the end of the section)

    When rasterizing line segments that could be sent to multiple viewports
    via the gl_ViewportMask[] built-in geometry shader output (section
    13.6.1), the line stipple pattern is not guaranteed to be continuous if
    segments are sent to multiple viewports.  If a line segment is not an
    independent line segment and is not the first in a series of connected
    segments (where the stipple counter <s> is reset to 0), the initial value
    of <s> for the segment is undefined unless that line segment and all
    previous segments in the series were sent to the same single viewport.

New Implementation Dependent State

    None.

New State

    None.

Additions to the AGL/GLX/WGL Specifications

    None.

GLX Protocol

    None.
    
Modifications to the OpenGL Shading Language Specification, Version 4.30

    Including the following line in a shader can be used to control the
    language features described in this extension:

      #extension GL_NV_viewport_array2 : <behavior>

    where <behavior> is as specified in section 3.3.

    New preprocessor #defines are added to the OpenGL Shading Language:

      #define GL_NV_viewport_array2                     1

    Modify Section 7.1 (Built-In Language Variables), p. 116

    Add to the list of vertex shader built-ins:

        out gl_PerVertex {
            highp int gl_ViewportIndex;
            highp int gl_ViewportMask[];
            highp int gl_Layer;
        };

    Add to the list of geometry shader built-ins:

        out highp int gl_ViewportMask[];

    Add to the list of tessellation control shader built-ins:

        out gl_PerVertex {
            highp int gl_ViewportIndex;
            highp int gl_ViewportMask[];
            highp int gl_Layer;
        } gl_out[];

    Add to the list of tessellation evaluation shader built-ins:

        out gl_PerVertex {
            highp int gl_ViewportIndex;
            highp int gl_ViewportMask[];
            highp int gl_Layer;
        };

    Modify descriptions of gl_Layer and gl_ViewportIndex as follows:

    The variable gl_Layer is available as an output variable in the vertex, 
    tessellation, and geometry (VTG) languages and an input variable in the 
    fragment language. In the VTG languages, it is used to select a specific 
    layer (or face and layer of a cube map) of a multi-layer framebuffer 
    attachment. The actual layer used will come from one of the vertices in 
    the primitive being shaded. Which vertex the layer comes from is undefined,
    so it is best to write the same layer value for all vertices of a 
    primitive. If a shader statically assigns a value to gl_Layer, layered 
    rendering mode is enabled. See section 11.4 "Layer and Viewport Selection" 
    and section 9.4.9 "Layered Framebuffers" of the OpenGL Graphics System 
    Specification for more information. If a shader statically assigns a value 
    to gl_Layer, and there is an execution path through the shader that does 
    not set gl_Layer, then the value of gl_Layer is undefined for executions 
    of the shader that take that path.

    ...

    The input variable gl_Layer in the fragment language will have the same 
    value that was written to the output variable gl_Layer in the VTG 
    languages. If the final VTG stage does not dynamically assign a value to 
    gl_Layer, the value of gl_Layer in the fragment stage will be undefined. 
    If the final VTG stage makes no static assignment to gl_Layer, the input 
    gl_Value in the fragment stage will be zero. Otherwise, the fragment stage 
    will read the same value written by the final VTG stage, even if that value
    is out of range. If a fragment shader contains a static access to gl_Layer,
    it will count against the implementation defined limit for the maximum 
    number of inputs to the fragment stage.

    The variable gl_ViewportIndex is available as an output variable in the 
    VTG languages and an input variable in the fragment language. In the 
    geometry language, it provides the index of the viewport to which the next 
    primitive emitted from the geometry shader should be drawn. In the vertex 
    and tessellation languages, it provides the index of the viewport 
    associated with the vertex being shaded. Primitives will undergo viewport 
    transformation and scissor testing using the viewport transformation and 
    scissor rectangle selected by the value of gl_ViewportIndex. The viewport 
    index used will come from one of the vertices in the primitive being 
    shaded. However, which vertex the viewport index comes from is 
    implementation-dependent, so it is best to use the same viewport index for 
    all vertices of the primitive. If the final VTG stage does not assign a 
    value to gl_ViewportIndex or gl_ViewportMask[], viewport transform and 
    scissor rectangle zero will be used. If a shader statically assigns a value
    to gl_ViewportIndex and there is a path through the shader that does not 
    assign a value to gl_ViewportIndex, the value of gl_ViewportIndex is 
    undefined for executions of the shader that take that path. See section 
    11.4 "Layer and Viewport Selection" of the OpenGL Graphics System 
    Specification for more information.

    The input variable gl_ViewportIndex in the fragment stage will have the 
    same value that was written to the output variable gl_ViewportIndex in the 
    final VTG stage. If the final VTG stage does not dynamically assign to 
    gl_ViewportIndex, the value of gl_ViewportIndex in the fragment shader will
    be undefined. If the final VTG stage makes no static assignment to 
    gl_ViewportIndex, the fragment stage will read zero. Otherwise, the 
    fragment stage will read the same value written by the final VTG stage, 
    even if that value is out of range. If a fragment shader contains a static 
    access to gl_ViewportIndex, it will count against the implementation 
    defined limit for the maximum number of inputs to the fragment stage.

    The variable gl_ViewportMask[] is available as an output variable in the
    VTG languages. The array has ceil(v/32) elements where v is the maximum
    number of viewports supported by the implementation. When a shader writes
    this variable, bit B of element M controls whether the primitive is
    emitted to viewport 32*M+B. If gl_ViewportIndex is written by the final
    VTG stage, then gl_ViewportIndex in the fragment stage will have the same
    value. If gl_ViewportMask[] is written by the final VTG stage, then
    gl_ViewportIndex in the fragment stage will have the index of the viewport
    that was used in generating that fragment.

    If a shader statically assigns a value to gl_ViewportIndex, it may not
    assign a value to any element of gl_ViewportMask[]. If a shader
    statically writes a value to any element of gl_ViewportMask[], it may
    not assign a value to gl_ViewportIndex. That is, a shader may assign
    values to either gl_ViewportIndex or gl_ViewportMask[], but not
    both. Multiple shaders linked together must also consistently write just
    one of these variables.  These incorrect usages all generate compile-time
    or link-time errors.

    The shader output gl_Layer may be redeclared with a layout qualifer 
    <viewport_relative> as follows:

        layout (viewport_relative) out highp int gl_Layer;

    If gl_Layer is <viewport_relative>, then the viewport index is added to
    the layer used for rendering (and available in the fragment shader). If
    the shader writes gl_ViewportMask[], then gl_Layer has a different value
    for each viewport the primitive is rendered to. If gl_Layer is
    <viewport_relative> and the shader writes neither gl_ViewportIndex nor
    gl_ViewportMask[], a link-error will result.

    Modify Section 8.15 (Geometry Shader Functions)

    The function EmitStreamVertex() specifies that a vertex is completed. A
    vertex is added to the current output primitive in vertex stream <stream>
    using the current values of all output variables associated with <stream>.
    These include gl_PointSize, gl_ClipDistance, gl_Layer, gl_Position,
    gl_PrimitiveID, gl_ViewportIndex, and gl_ViewportMask[]. The values of
    all output variables for all output streams are undefined after a call to
    EmitStreamVertex().

Errors

    None.

Interactions with OpenGL ES 3.1

    Unless functionality similar to ARB_provoking_vertex is supported, remove
    references to PROVOKING_VERTEX and ProvokingVertex().  Also remove
    reference to 'vertex color clamping'.  The modifications to Line Stippling
    don't apply.

Interactions with EXT_tessellation_shader

    If implemented on OpenGL ES and EXT_tessellation_shader is not supported,
    remove all language referring to the tesselation control and tessellation
    evaluation pipeline stages.

Interactions with NV_geometry_shader_passthrough

    If NV_geometry_shader_passthrough is supported, the NV_gpu_program4 and
    NV_geometry_program4 language describing the PASSTHROUGH declaration
    statement should be modified to state that "result.viewportmask" may not
    be used in such a declaration.

Interactions with NV_gpu_program4

    If NV_gpu_program4 is supported and the "NV_viewport_array2" program
    option is specified, vertex, tessellation control/evaluation, and geometry 
    program result variable "result.viewportmask" can be used to specify the 
    mask of viewports that the primitive will be emitted to, "result.viewport"
    can be used to specify the index of the viewport that the primitive will
    be emitted to, and "result.layer" can be used to specify the layer of a
    layered framebuffer attachment that the primitive will be emitted to.

    (add the following rule to the NV_gpu_program4 grammar)

    <resultBasic>      ::= ...
                         | <resPrefix> "viewportmask" arrayMemAbs
                         | <resPrefix> "viewport"
                         | <resPrefix> "layer"

    (add the following to the tables of Vertex, Geometry, and Tessellation 
    Control/Eval Program Result Variable Bindings)

      Binding                        Components  Description
      -----------------------------  ----------  ----------------------------
      result.viewportmask[]          (v,*,*,*)   viewport array mask
      result.viewport                (v,*,*,*)   viewport array index
      result.layer                   (l,*,*,*)   layer for cube/array/3D FBOs

    (add the following to Section 2.X.2, Program Grammar)

    If a result variable binding matches "result.viewportmask", updates to the 
    "x" component of the result variable provide a single integer that serves 
    as a mask of viewport indices. The mask must be written as an integer 
    value; writing a floating-point value will produce undefined results. 
    If the value has bits greater than or equal to MAX_VIEWPORTS set, the
    number of viewports the primitive is emitted to and which viewports are 
    used are undefined. If the "NV_viewport_array2" program option is not 
    specified, the "result.viewportmask" binding is unavailable.

    If both "result.viewport" and "result.viewportmask" are written, 
    compilation will fail.

    (add the following to Section 2.X.6.Y, Program Options)

    + Viewport Mask (NV_viewport_array2)

    If a vertex, geometry, tessellation control, or tessellation evaluation 
    program specifies the "NV_viewport_array2" option, the result binding 
    "result.viewportmask" will be available to specify the mask of viewports
    to use for primitive viewport transformations and scissoring as described 
    in section 2.X.2. Additionally, the "result.viewport" and "result.layer"
    result bindings will be available in these same shader stages.

    If a program specifies the "NV_layer_viewport_relative" option, the 
    result.layer will have the viewport index automatically added to it. If
    the result.viewportmask is used, the result.layer will be different for 
    each viewport the primitive is emitted to.

Issues

    (1) Where does the viewport mask broadcast occur?

    RESOLVED:  This operation could potentially be performed before or after
    transform feedback, but feeding back several viewports worth of primitives
    doesn't seem particularly useful.  This specification applies the viewport
    mask after transform feedback, and makes primitive queries only count each
    primitive once.

    Note that it is possible to capture viewport mask shader outputs when
    transform feedback is active.

    (2) How does the gl_ViewportIndex fragment input behave?

    RESOLVED:  Whether viewport mask or viewport indices are used in VTG
    shaders, the fragment shader input gl_ViewportIndex will contain the
    viewport number for the primitive generating each fragment.  If the
    viewport mask is used to broadcast a single primitive to multiple
    viewports, and the same pixel is covered by the primitive in each
    viewport, multiple fragment shader invocations for that pixel will be
    generated, each with a different value of gl_ViewportIndex.

    This extension provides no gl_ViewportMask[] input, so a fragment shader
    is not able to see the original viewport mask for the primitive generating
    the fragment.  If necessary, this value could be passed by a separate
    shader variable qualified with "flat".

    (3) How does the viewport mask interact with line stipple?

    RESOLVED:  With viewport mask, it's possible to broadcast line strips to
    multiple viewports.  If line stipple is enabled in the OpenGL
    compatibility profile, implementations are required to maintain a
    continuous stipple pattern across the strip.  When primitives are
    broadcast via viewport mask, implementations will not always be able to
    buffer an entire strip and send it to each viewport in turn.  So it will
    often be necessary to break up a long strip, and send segments to
    alternating viewports.  An implementation could handle this by breaking up
    the strip and keeping N independent stipple counters, but that seems like
    overkill.

    We relax normal spec requirements and require a continuous stipple pattern
    only if the entire strip if sent to exactly one viewport.  If any segment
    in the strip is sent to multiple viewports, no viewports, or a different
    viewport than previous segments, the stipple counter for that segment and
    subsequent segments in the strip is undefined.

    (4) Can a viewport index or mask written by vertex or tessellation shader
    be read by downstream tessellation or geometry shaders?

    RESOLVED:  No. The fragment shader is able to read the viewport index, but
    this extension provides no built-in input allowing VTG shaders to see a
    viewport index or mask written by a previous shader stage.

    (5) Can this extension be used to "kill" primitives in a passthrough
    geometry shader (NV_geometry_shader_passthrough)?

    RESOLVED:  Yes.  In regular geometry shaders, input primitives can be
    killed by returning without emitting any vertices.  That's not possible
    with passthrough geometry shaders, however it is possible to code a
    passthrough geometry shader like:

      void main()
      {
        if (shouldKillPrimitive()) {
          // Set the viewport mask to zero.  A primitive will still be
          // emitted from the geometry shader stage, however it will be sent
          // to no viewports and thus be discarded.  Any other per-primitive
          // outputs will be undefined.
          gl_ViewportMask[0] = 0;
          return;
        }

        // Since the shader writes the viewport mask in the "kill" path, it
        // also needs to write it in the non-"kill" path; otherwise, its
        // value would be undefined and the primitive will be sent to an
        // undefined set of viewports.  Setting the mask to 1 will always
        // send a primitive to viewport zero.
        gl_ViewportMask[0] = 1;

        ...
      }

    Without the viewport mask or a similar feature, it is not possible to kill
    primitives in a passthrough geometry shader.

Revision History

    Revision 2, 2015/03/27
      - Add ES interactions

    Revision 1
      - Internal revisions.
