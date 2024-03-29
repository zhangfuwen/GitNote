# NV_viewport_swizzle

Name

    NV_viewport_swizzle

Name Strings

    GL_NV_viewport_swizzle

Contact

    Jeff Bolz, NVIDIA Corporation (jbolz 'at' nvidia.com)
    Pat Brown, NVIDIA Corporation (pbrown 'at' nvidia.com)

Contributors

    Mathias Heyer, NVIDIA

Status

    Shipping.

Version

    Last Modified Date:         April 7, 2015
    Revision:                   1

Number

    OpenGL Extension #483
    OpenGL ES Extension #258

Dependencies

    This extension is written against the OpenGL 4.3 specification
    (Compatibility Profile).

    This extension interacts with the OpenGL ES 3.1 (March 17, 2014)
    specification.

    This extension interacts with NV_viewport_array2.

Overview

    This extension provides a new per-viewport swizzle that can modify the
    position of primitives sent to each viewport.  New viewport swizzle state
    is added for each viewport, and a new position vector is computed for each
    vertex by selecting from and optionally negating any of the four
    components of the original position vector.

    This new viewport swizzle is useful for a number of algorithms, including
    single-pass cubemap rendering (broadcasting a primitive to multiple faces
    and reorienting the vertex position for each face) and voxel
    rasterization.  The per-viewport component remapping and negation provided
    by the swizzle allows application code to re-orient three-dimensional
    geometry with a view along any of the X, Y, or Z axes.  If a perspective
    projection and depth buffering is required, 1/W buffering should be used,
    as described in the single-pass cubemap rendering example in the "Issues"
    section below.

New Procedures and Functions

    void ViewportSwizzleNV(uint index,
                           enum swizzlex, enum swizzley,
                           enum swizzlez, enum swizzlew);

New Tokens

    Accepted by the <swizzlex>, <swizzley>, <swizzlez>, and <swizzlew>
    parameters of ViewportSwizzleNV:

        VIEWPORT_SWIZZLE_POSITIVE_X_NV                  0x9350
        VIEWPORT_SWIZZLE_NEGATIVE_X_NV                  0x9351
        VIEWPORT_SWIZZLE_POSITIVE_Y_NV                  0x9352
        VIEWPORT_SWIZZLE_NEGATIVE_Y_NV                  0x9353
        VIEWPORT_SWIZZLE_POSITIVE_Z_NV                  0x9354
        VIEWPORT_SWIZZLE_NEGATIVE_Z_NV                  0x9355
        VIEWPORT_SWIZZLE_POSITIVE_W_NV                  0x9356
        VIEWPORT_SWIZZLE_NEGATIVE_W_NV                  0x9357

    Accepted by the <pname> parameter of GetBooleani_v, GetDoublei_v,
    GetIntegeri_v, GetFloati_v, and GetInteger64i_v:

        VIEWPORT_SWIZZLE_X_NV                           0x9358
        VIEWPORT_SWIZZLE_Y_NV                           0x9359
        VIEWPORT_SWIZZLE_Z_NV                           0x935A
        VIEWPORT_SWIZZLE_W_NV                           0x935B

Additions to Chapter 13 of the OpenGL 4.3 (Compatibility Profile)
Specification (Fixed-Function Vertex Post-Processing)

    Modify Section 13.2 (Transform Feedback), p. 453

    Modify the first paragraph:

    ...The vertices are fed back after vertex color clamping, but before
    viewport swizzling and viewport mask expansion, flatshading, and
    clipping. ...


    Add a new Section 13.X (Viewport Swizzle) after 13.3 (Primitive Queries)

    Each primitive sent to a given viewport has a swizzle and optional
    negation applied to its clip coordinates.  The swizzle that is applied
    depends on the viewport index, and is controlled by the command

        void ViewportSwizzleNV(uint index,
                               enum swizzlex, enum swizzley,
                               enum swizzlez, enum swizzlew);

    The viewport specified by <index> has its x,y,z,w swizzle state set to the
    corresponding <swizzlex>, <swizzley>, <swizzlez>, <swizzlew> value. If the
    value of VIEWPORT_SWIZZLE_X_NV is denoted by <swizzlex>, swizzling computes
    the new x component of the position as

        if (swizzlex == VIEWPORT_SWIZZLE_POSITIVE_X_NV) x' = x;
        if (swizzlex == VIEWPORT_SWIZZLE_NEGATIVE_X_NV) x' = -x;
        if (swizzlex == VIEWPORT_SWIZZLE_POSITIVE_Y_NV) x' = y;
        if (swizzlex == VIEWPORT_SWIZZLE_NEGATIVE_Y_NV) x' = -y;
        if (swizzlex == VIEWPORT_SWIZZLE_POSITIVE_Z_NV) x' = z;
        if (swizzlex == VIEWPORT_SWIZZLE_NEGATIVE_Z_NV) x' = -z;
        if (swizzlex == VIEWPORT_SWIZZLE_POSITIVE_W_NV) x' = w;
        if (swizzlex == VIEWPORT_SWIZZLE_NEGATIVE_W_NV) x' = -w;

    Similar selections are performed for the y, z, and w coordinates. This
    swizzling is applied after transform feedback, but before clipping and
    perspective divide.

    Errors:

    - The error INVALID_VALUE is generated if <index> is greater than or equal
      to the value of MAX_VIEWPORTS.

    - The error INVALID_ENUM is generated if any of <swizzlex>, <swizzley>,
      <swizzlez>, or <swizzlew> are not one of
      VIEWPORT_SWIZZLE_{POSITIVE,NEGATIVE}_{X,Y,Z,W}.


    Modify Section 13.6.1 (Controlling the Viewport)

    (modify the first paragraph, p. 470, as edited by NV_viewport_array2,
    using "transformed and swizzled" instead of "transformed")

    Multiple viewports are available ... The primitive is transformed and
    swizzled using the state of the selected viewport. ...

    ... If bit <i> is set in the mask, the primitive is emitted to viewport
    <i> and transformed and swizzled using the state of viewport <i>. ...


New Implementation Dependent State

    None.

New State

    Get Value                       Get Command    Type    Initial Value        Description                 Sec.    Attribute
    ---------                       -----------    ----    -------------        -----------                 ----    ---------
    VIEWPORT_SWIZZLE_X_NV           GetIntegeri_v  nxZ8    VIEWPORT_SWIZZLE-    coordinate and sign for     13.X    viewport
                                                           POSITIVE_X           viewport swizzling
    VIEWPORT_SWIZZLE_Y_NV           GetIntegeri_v  nxZ8    VIEWPORT_SWIZZLE-    coordinate and sign for     13.X    viewport
                                                           POSITIVE_Y           viewport swizzling
    VIEWPORT_SWIZZLE_Z_NV           GetIntegeri_v  nxZ8    VIEWPORT_SWIZZLE-    coordinate and sign for     13.X    viewport
                                                           POSITIVE_Z           viewport swizzling
    VIEWPORT_SWIZZLE_W_NV           GetIntegeri_v  nxZ8    VIEWPORT_SWIZZLE-    coordinate and sign for     13.X    viewport
                                                           POSITIVE_W           viewport swizzling

Additions to the AGL/GLX/WGL Specifications

    None.

GLX Protocol

    None.

Errors

    The error INVALID_VALUE is generated by ViewportSwizzleNV if <index> is
    greater than or equal to the value of MAX_VIEWPORTS.

    The error INVALID_ENUM is generated by ViewportSwizzleNV if any of
    <swizzlex>, <swizzley>, <swizzlez>, or <swizzlew> are not one of
    VIEWPORT_SWIZZLE_{POSITIVE,NEGATIVE}_{X,Y,Z,W}.

Interactions with OpenGL ES 3.1

    Remove references to GetDoublei_v and GetBooleani_v.  Also remove the
    reference to 'vertex color clamping'.

Interactions with NV_viewport_array2

    This specification modifies language added/changed by NV_viewport_array2.
    There are no functional interactions between the two extensions, though we
    expect that all implementations of this extension will support
    NV_viewport_array2 or similar functionality.

Issues

    (1) Where does viewport swizzling occur in the pipeline?

    RESOLVED: Despite being associated with the viewport, viewport swizzling
    must happen prior to the viewport transform.  In particular, it needs to
    be performed before clipping and perspective division.

    The viewport mask expansion (NV_viewport_array2) and the viewport swizzle
    could potentially be performed before or after transform feedback, but
    feeding back several viewports worth of primitives with different swizzles
    doesn't seem particularly useful.  This specification applies the viewport
    mask and swizzle after transform feedback, and makes primitive queries
    only count each primitive once.

    (2) Any interesting examples of how this extension, NV_viewport_array2,
    and NV_geometry_shader_passthrough can be used together in practice?

    RESOLVED:  One interesting use case for this extension is for single-pass
    rendering to a cubemap.  In this example, the application would attach a
    cubemap texture to a layered FBO where the six cube faces are treated as
    layers.  Vertices are sent through the vertex shader without applying a
    projection matrix, where the gl_Position output is (x,y,z,1) and the
    center of the cubemap is at (0,0,0).  With unextended OpenGL, one could
    have a conventional instanced geometry shader that looks something like
    the following:

      layout(invocations = 6) in;     // separate invocation per face
      layout(triangles) in;
      layout(triangle_strip) out;
      layout(max_vertices = 3) out;

      in Inputs {
        vec2 texcoord;
        vec3 normal;
        vec4 baseColor;
      } v[];

      out Outputs {
        vec2 texcoord;
        vec3 normal;
        vec4 baseColor;
      };

      void main()
      {
        int face = gl_InvocationID;  // which face am I?

        // Project gl_Position for each vertex onto the cube map face.
        vec4 positions[3];
        for (int i = 0; i < 3; i++) {
          positions[i] = rotate(gl_in[i].gl_Position, face);
        }

        // If the primitive doesn't project onto this face, we're done.
        if (shouldCull(positions)) {
          return;
        }

        // Otherwise, emit a copy of the input primitive to the
        // appropriate face (using gl_Layer).
        for (int i = 0; i < 3; i++) {
          gl_Layer = face;
          gl_Position = positions[i];
          texcoord = v[i].texcoord;
          normal = v[i].normal;
          baseColor = v[i].baseColor;
          EmitVertex();
        }
      }

    With passthrough geometry shaders, this can be done using a much simpler
    shader:

      layout(triangles) in;
      layout(passthrough) in Inputs {
        vec2 texcoord;
        vec3 normal;
        vec4 baseColor;
      }
      layout(passthrough) in gl_PerVertex {
        vec4 gl_Position;
      } gl_in[];
      layout(viewport_relative) out int gl_Layer;

      void main()
      {
        // Figure out which faces the primitive projects onto and
        // generate a corresponding viewport mask.
        uint mask = 0;
        for (int i = 0; i < 6; i++) {
          if (!shouldCull(face)) {
            mask |= 1U << i;
          }
        }
        gl_ViewportMask = mask;
        gl_Layer = 0;
      }

    The application code is set up so that each of the six cube faces has a
    separate viewport (numbered 0..5).  Each face also has a separate swizzle,
    programmed via the ViewportSwizzleNV() command.  The viewport swizzle
    feature performs the coordinate transformation handled by the rotate()
    function in the original shader.  The "viewport_relative" layout qualifier
    says that the viewport number (0..5) is added to the base gl_Layer value
    of zero to determine which layer (cube face) the primitive should be sent
    to.

    Note that the use of the passed through input <normal> in this example
    suggests that the fragment shader in this example would perform an
    operation like per-fragment lighting.  The viewport swizzle would
    transform the position to be face-relative, but <normal> would remain in
    the original coordinate system.  It seems likely that the fragment shader
    in either version of the example would want to perform lighting in the
    original coordinate system.  It would likely do this by reconstructing the
    position of the fragment in the original coordinate system using
    gl_FragCoord, a constant or uniform holding the size of the cube face, and
    the input gl_ViewportIndex (or gl_Layer), which identifies the cube face.
    Since the value of <normal> is in the original coordinate system, it would
    not need to be modified as part of this coordinate transformation.

    Note that while the rotate() operation in the regular geometry shader
    above could include an arbitrary post-rotation projection matrix, the
    viewport swizzle does not support arbitrary math.  To get proper
    projection, 1/W buffering should be used.  To do this:

      (1) Program the viewport swizzles to move the pre-projection W eye
      coordinate (typically 1.0) into the Z coordinate of the swizzle output
      and the eye coordinate component used for depth into the W coordinate.
      For example, the viewport corresponding to the +Z face might use a
      swizzle of (+X, -Y, +W, +Z).  The Z normalized device coordinate
      computed after swizzling would then be z'/w' = 1/Z_eye.

      (2a) On NVIDIA implementations supporting floating-point depth buffers
      with values outside [0,1], prevent unwanted near plane clipping by
      enabling DEPTH_CLAMP.  Ensure that the depth clamp doesn't mess up depth
      testing by programming the depth range to very large values, such as
      glDepthRangedNV(-z, +z), where z == 2^127.  It should be possible to use
      IEEE infinity encodings also (0xFF800000 for -INF, 0x7F800000 for +INF).
      Even when near/far clipping is disabled, primitives extending behind the
      eye will still be clipped because one or more vertices will have a
      negative W coordinate and fail X/Y clipping tests.

      (2b) On other implementations, scale X, Y, and Z eye coordinates so that
      vertices on the near plane have a post-swizzle W coordinate of 1.0.  For
      example, if the near plane is at Z_eye = 1/256, scale X, Y, and Z by
      256.  Also, ideally, program the depth range transformation to be a NOP
      by using a clip control depth mode (OpenGL 4.5) of ZERO_TO_ONE.

      (3) Adjust depth testing to reflect the fact that 1/W values are large
      near the eye and small away from the eye.  Clear the depth buffer to
      zero (infinitely far away) and use a depth test of GREATER instead of
      LESS.

Revision History

    Revision 1
    - Internal revisions.
