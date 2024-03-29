# OES_primitive_bounding_box

Name

    OES_primitive_bounding_box

Name Strings

    GL_OES_primitive_bounding box

Contact

    Jesse Hall (jessehall 'at' google.com)

Contributors

    Alex Chalfin, ARM
    Jan-Harald Fredriksen, ARM
    Dan Stoza, Google
    Cass Everitt, NVIDIA
    Daniel Koch, NVIDIA
    Jeff Bolz, NVIDIA
    Pat Brown, NVIDIA
    Bill Licea-Kane, Qualcomm
    Maurice Ribble, Qualcomm
    Vineet Goel, Qualcomm

Notice

    Copyright (c) 2014-2018 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Specification Update Policy

    Khronos-approved extension specifications are updated in response to
    issues and bugs prioritized by the Khronos OpenGL ES Working Group. For
    extensions which have been promoted to a core Specification, fixes will
    first appear in the latest version of that core Specification, and will
    eventually be backported to the extension document. This policy is
    described in more detail at
        https://www.khronos.org/registry/OpenGL/docs/update_policy.php

Status

    Approved by the OpenGL ES Working Group
    Ratified by the Khronos Board of Promoters on November 7, 2014

Version

    Last Modified Date: February 13, 2018
    Revision: 3

Number

    OpenGL ES Extension #212

Dependencies

    OpenGL ES 3.1 and OpenGL ES Shading Language 3.10 are required.

    This specification is written against the OpenGL ES 3.1 (March 17,
    2014) and OpenGL ES 3.10 Shading Language (March 17, 2014)
    Specifications.

    This extension interacts with the OpenGL ES 3.20 Shading Language.

    This extension interacts with EXT_tessellation_shader and
    OES_tessellation_shader.

    OES_geometry_shader and EXT_geometry_shader trivially affect the
    definition of this extension.

Overview

    On tile-based architectures, transformed primitives are generally written
    out to memory before rasterization, and then read back from memory for each
    tile they intersect. When geometry expands during transformation due to
    tessellation or one-to-many geometry shaders, the external bandwidth
    required grows proportionally. This extension provides a way for
    implementations to know which tiles incoming geometry will intersect before
    fully transforming (and expanding) the geometry. This allows them to only
    store the unexpanded geometry in memory, and perform expansion on-chip for
    each intersected tile.

    New state is added to hold an axis-aligned bounding box which is assumed to
    contain any geometry submitted. An implementation is allowed to ignore any
    portions of primitives outside the bounding box; thus if a primitive extends
    outside of a tile into a neighboring tile that the bounding box didn't
    intersect, the implementation is not required to render those portions. The
    tessellation control shader is optionally able to specify a per-patch
    bounding box that overrides the bounding box state for primitives generated
    from that patch, in order to provide tighter bounds.

    The typical usage is that an application will have an object-space bounding
    volume for a primitive or group of primitives, either calculated at runtime
    or provided with the mesh by the authoring tool or content pipeline. It will
    transform this volume to clip space, compute an axis-aligned bounding box
    that contains the transformed bounding volume, and provide that at either
    per-patch or per-draw granularity.

IP Status

    No known IP claims.

New Procedures and Functions

    void PrimitiveBoundingBoxOES(float minX, float minY, float minZ, float minW,
                                 float maxX, float maxY, float maxZ, float maxW);

New Tokens

    Accepted by the <pname> parameter of GetBooleanv, GetFloatv, GetIntegerv,
    and GetInteger64v:

        PRIMITIVE_BOUNDING_BOX_OES                          0x92BE

Additions to the OpenGL ES 3.1 Specification

    Modify section 11.1ts.1.2.3, "Tessellation Control Shader Outputs", as added
    by OES_tessellation_shader or EXT_tessellation_shader:

    In the second paragraph, add gl_BoundingBoxOES[] to the list of built-in
    per-patch output arrays:

    Tessellation shaders additionally have three built-in per-patch output
    arrays, gl_TessLevelOuter[], gl_TessLevelInner[], and gl_BoundingBoxOES[].
    These arrays ... as discussed in the following section. gl_BoundingBoxOES[]
    is an array of two vec4 values that should be used instead of the value of
    PRIMITIVE_BOUNDING_BOX_OES as the primitive bounding box (see Section
    13.1pbb) for primitives generated from the output patch.

    Modify the sixth paragraph of the section to state that gl_BoundingBoxOES[]
    counts against the per-patch limit:

    ... The built-in outputs gl_TessLevelOuter[] and gl_TessLevelInner[] are not
    counted against the per-patch limit. The built-in output
    gl_BoundingBoxOES[], if statically assigned by the shader, is counted
    against the per-patch limit. The total number of components...


    Modify section 11.1ts.3.3, "Tessellation Evaluation Shader Inputs", as added
    by OES_tessellation_shader or EXT_tessellation_shader:

    Insert a new paragraph after the list of special input variables in
    paragraph 2:

    The special tessellation control shader output gl_BoundingBoxOES[] is
    consumed by the tessellation primitive generator, and is not available as an
    input to the tessellation evaluation shader.


    Add new section 13.1pbb following section 13.1, "Discarding Primitives
    Before Rasterization" on p. 288:

    13.1pbb, Primitive Bounding Box

    Implementations may be able to optimize performance if the application
    provides bounds of primitives that will be generated by the tessellation
    primitive generator or the geometry shader prior to executing those stages.
    If the provided bounds are incorrect and primitives extend beyond them, the
    rasterizer may or may not generate fragments for the portions of primitives
    outside the bounds.

    The primitive bounding box is specified using

        void PrimitiveBoundingBoxOES(float minX, float minY, float minZ, float minW,
                                     float maxX, float maxY, float maxZ, float maxW);

    where <minX>, <minY>, <minZ>, and <minW> specify the minimum clip space
    coordinate of the bounding box and <maxX>, <maxY>, <maxZ>, and <maxW>
    specify the maximum coordinate.

    If tessellation is active, each invocation of the tessellation control
    shader may re-specify the bounding box by writing to the built-in
    gl_BoundingBoxOES[] variable. If the shader statically assigns a value to
    any part of this variable, then gl_BoundingBoxOES[0] is used instead of
    <minX>, <minY>, <minZ>, <minW>, and gl_BoundingBoxOES[1] is used instead of
    <maxX>, <maxY>, <maxZ>, <maxW>.  If the shader contains a static assignment
    to gl_BoundingBoxOES[] and there is an execution path through the shader
    that does not write all components of gl_BoundingBoxOES[], the value of
    unwritten components and corresponding bounding box coordinates is undefined
    for executions of the shader that take that path.

    If the tessellation control shader re-specifies the bounding box, the re-
    specified value is used for primitives generated from the output patch by
    the primitive generator, any primitives emitted by the geometry shader
    invocations for those generated primitives, and any primitives further
    introduced during clipping.

    The bounding box in clip space is composed of 16 vertices formed by all
    combinations of the minimum and maximum values for each dimension. This
    bounding box is clipped against w_c > 0, and projected to three dimensions
    by dividing x_c, y_c, and z_c by w_c for each vertex. The viewport transform
    is then applied to each vertex to produce a three-dimensional bounding
    volume in window coordinates.

    The window space bounding volume is expanded in the X and Y dimensions to
    accomodate the rasterization rules for the primitive type, and to fall on
    fragment boundaries:
        min_wc' = floor(min_wc - size/2.0)
        max_wc' = ceil(max_wc + size/2.0)
    where the min_wc rule is used for x and y window coordinates of bounding
    volume vertices formed from minX and minY respectively, and the max_wc rule
    is used for x and y window coordinates of bounding volume vertices formed
    from maxX and maxY respectively. For point primitives, size is the per-
    primitive point size after clamping to the implementation-defined maximum
    point size as described in section 13.3. For line primitives, size is the
    line width, after rounding and clamping as described in section 13.4.2.1.
    For triangle primitives, size is zero.

    During rasterization, the rasterizer will generate fragments with
    window coordinates inside the windows space bounding volume, but may or may
    not generate fragments with window coordinates outside the bounding volume.


Dependencies on OES_tessellation_shader and EXT_tessellation_shader

    If OES_tessellation_shader, EXT_tessellation_shader or equivalent
    functionality is not supported, ignore all references to the
    gl_BoundingBoxOES[] built-in per-patch variable in tessellation control and
    evaluation shaders, and remove references to the tessellation primitive
    generator.

Dependencies on OES_geometry_shader and EXT_geometry_shader

    If OES_geometry_shader, EXT_geometry_shader or equivalent functionality is
    not supported, remove all references to geometry shaders.
    OES_tessellation_shader requires OES_geometry_shader, if OES_geometry_shader
    is not supported there is probably no benefit to supporting this extension.

New State

    Add to state values in Table 6.5, Transformation State:

                                                  Default
    Get Value                  Type  Get Command  Value        Description       Sec.
    -------------------------- ----  -----------  ------------ ----------------- --------
    PRIMITIVE_BOUNDING_BOX_OES 8xR   GetFloatv    -1,-1,-1, 1, Default primitive 13.1pbb
                                                   1, 1, 1, 1  bounding box


Additions to the OpenGL ES Shading Language 3.10 Specification

    Including the following line in a shader can be used to control the
    language features described in this extension:

      #extension GL_OES_primitive_bounding_box : <behavior>

    where <behavior> is as specified in section 3.4.

    A new preprocessor #define is added to the OpenGL ES Shading Language:

      #define GL_OES_primitive_bounding_box 1


    Modify subsection 7.1ts1 "Tessellation Control Special Variables" as added
    by OES_tessellation_shader:

    Add a new built-in variable intrinsic declaration after the
    gl_TessLevelOuter and gl_TessLevelInner declarations:

      patch out highp vec4 gl_BoundingBoxOES[2];

    Add the following paragraph to subsection 7.1ts1.2 "Tessellation Control
    Output Variables" as added by OES_tessellation_shader, after the paragraph
    describing gl_TessLevelOuter and gl_TessLevelInner:

      The values written to gl_BoundingBoxOES specify the minimum and maximum
      clip-space extents of a bounding box containing all primitives generated
      from the patch by the primitive generator, geometry shader, and clipping.
      Fragments may or may not be generated for portions of these primitives
      that extend outside the window-coordinate projection of this bounding
      box.


Dependencies on the OpenGL ES 3.20 Shading Language Specification

    If GL_OES_primitive_bounding_box is enabled in an OpenGL ES 3.20 shader,
    gl_BoundingBoxOES is an alias for gl_BoundingBox: the last value written
    to either variable will be used for bounding box optimizations and for
    reads from both variables.


Issues

    (1) What coordinate space should the bounding box be specified in?

    RESOLVED: The bounding box is specified in clip coordinates.

    Using clip coordinates is consistent with vertex positions, allowing apps
    to use the same transformations. Another option was NDC, but NDC is not
    used directly by application or shader code anywhere else in the API, and
    many developers are unfamiliar with it. Using clip coordinates gives
    implementations the maximum amount of flexibility in how they implement the
    subsequent transforms and bounding box test.

    (2) Should the bounds be 3D or 2D? The goal of optimizing tiling
    architectures only appears to need 2D.

    RESOLVED: The bounds are 3D. Depth bounds could be useful for culling
    patches prior to primitive generation if all the generated primitives would
    fail the depth test.

    (3) With the bounding box in clip coordinates, what happens when w <= 0?

    RESOLVED: Clip the bounding box against the w > 0 plane.

    This effectively pushes the front face of the bounding box forward until it
    is just in front of the camera. It is relatively simple in that no new
    vertices will be introduced and no connectivity will change.

    Note that, as with Issue #5, since false-positives are allowed, an
    implementation may be able to avoid this clipping altogether. As an extreme
    example, if either min or max w is <= 0, the implementation could disable
    bounding box optimizations.

    (4) What happens if a primitive extends outside the bounding box? Is
    behavior undefined for the whole primitive, or only for the portions outside
    the bounds? What restrictions can we place on the undefined behavior?

    RESOLVED: In the interest of limiting the scope of undefined behavior as
    much as possible without requiring precise clipping or scissoring, specify
    that portions of the primitive inside the bounding box generate fragments
    normally, and that the rasterizer may or may not generate fragments for
    portions of the primitive outside the bounding box.

    (5) What space should the bounding box test happen in, and how should it
    be specified?

    RESOLVED: The proposed resolution to Issue #4 is that fragments outside the
    bounds may or may not be generated. That makes window space the most natural
    place to specify the test. If that issue were resolved differently,
    specifying the bounds test in NDC space might have been simpler and more
    natural.

    In practice, implementations will probably do the test conservatively at an
    earlier stage than rasterization.  Because false-positives are allowed,
    implementations have a lot of flexibility in where and how precisely they
    perform the test, so the primary requirement for this spec is to provide
    useful and understandable guarantees about what parts of the primitive
    definitely will be rendered.

    (6) Should the bounding box test apply even when tessellation isn't active?

    RESOLVED: Yes. Having it apply in VS+GS+FS pipelines is useful to allow
    implementations to run the GS after binning, since like tessellation it can
    greatly magnify the amount of geometry. Unlike tessellation, because the
    expansion happens in the geometry shader, there isn't a natural way for the
    geometry shader itself to specify the bounding box for its own output
    primitives.

    Because the bounding box is no longer a tessellation-specific feature, it
    is now a separate extension from OES_tessellation_shader.

    (7) What should the initial value of the bounding box state be?

    RESOLVED: The initial value is {(-1,-1,-1,1), (1,1,1,1)}. This corresponds
    to the entire view frustum, so by default the bounding box test won't reject
    any fragments that would otherwise have been drawn.

    Another proposed option was to use positive and negative infinity values;
    that has no benefits over positive and negative one, and infinity values can
    be hard to generate in C code if an application wanted to return to the
    initial state.

    (8) Do we really need a 16-vertex bounding volume? Do we need the vertices
    with (minZ, maxW) or (maxZ, minW)?

    RESOLVED: Specify 16 vertices for generality.  Implementations may be able
    to avoid forming or transforming all of them explicitly.

    With standard orthographic (w==1) or perspective projections (such as those
    produced by glFrustum) and "normal" vertices, an eight-vertex bounding
    volume is sufficient because of the restricted relationships between z_c and
    w_c.  However, OpenGL ES does not require this restricted relationship, and
    in the general case all 16 vertices are needed. It's possible that
    applications attempting to bloat the bounding box to account for
    displacement mapping, gl_FragDepth modifications, etc. may break the
    standard relationship and require all 16 vertices.

    (9) Do we need a way to bloat the bounding volume in window space to account
    for things like point size, line width, etc.? Applying this bloat in clip
    space may be difficult or overly conservative.

    RESOLVED: Automatically expand the bounding volume in window space by the
    point size or line width.

    Another option considered was to add state for the application to specify a
    window-space expansion. This would impose an extra burden on the application
    to update the state to match line width or maximum point size being
    rendered, and strongly constrains implementations.

    (10) Should the TES be able to read gl_BoundingBoxOES[]?

    RESOLVED: No. The bounding box is consumed by the primitive generator.

    Being able to read gl_BoundingBoxOES[] in the TES doesn't seem particularly
    useful. It raises the question of what value it gets when the TCS doesn't
    write the bounding box (undefined, or the PRIMITIVE_BOUNDING_BOX_OES
    state?), and whether the TES is required to declare it when the TCS and TES
    are in different program objects. More importantly, reading the bounding box
    in the TES may impose surprising costs on some implementations where it is
    consumed by the fixed-function primitive generation stage.


Revision History

    Rev.    Date      Author    Changes
    ----  ----------  --------- -------------------------------------------------
     3    02/13/2018  jessehall Added a description of gl_BoundingBoxOES to the
                                shading language specification.

     2    09/07/2016  jessehall Added interaction with the OpenGL ES 3.20 Shading
                                Language.

     1    07/14/2014  dkoch     Initial OES version based on EXT.
                                No functional changes.
