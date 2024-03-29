# NV_fragment_shader_barycentric

Name

    NV_fragment_shader_barycentric

Name Strings

    GL_NV_fragment_shader_barycentric

Contact

    Pat Brown, NVIDIA (pbrown 'at' nvidia.com)

Contributors

    Ashwin Lele, NVIDIA
    Jeff Bolz, NVIDIA
    Michael Chock, NVIDIA

Status

    Shipping

Version

    Last Modified:      April 8, 2018
    Revision:           2

Number

    OpenGL Extension #526
    OpenGL ES Extension #316

Dependencies

    This extension is written against the OpenGL 4.6 Specification
    (Compatibility Profile), dated July 30, 2017.

    OpenGL 4.5 or OpenGL ES 3.2 is required.

    This extension requires support for the OpenGL Shading Language (GLSL)
    extension "NV_fragment_shader_barycentric", which can be found at the
    Khronos Group Github site here:

        https://github.com/KhronosGroup/GLSL

Overview

    This extension advertises OpenGL support for the OpenGL Shading Language
    (GLSL) extension "NV_fragment_shader_barycentric", which provides fragment
    shader built-in variables holding barycentric weight vectors that identify
    the location of the fragment within its primitive.  Additionally, the GLSL
    extension allows fragment the ability to read raw attribute values for
    each of the vertices of the primitive that produced the fragment.

New Procedures and Functions

    None

New Tokens

    None

Modifications to the OpenGL 4.6 Specification (Compatibility Profile)

    Modify Section 15.2.2, Shader Inputs (p. 586)

    (insert new paragraphs after the first paragraph, p. 589)

    Fragment shader input variables can be declared as per-vertex inputs using
    the GLSL interpolation qualifier "pervertexNV".  Such inputs are not
    produced by attribute interpolation, but are instead taken directly from
    corresponding output variables written by the previous shader stage, prior
    to primitive clipping and rasterization.  When reading per-vertex inputs,
    a fragment shader specifies a vertex number (0, 1, or 2) that identifies a
    specific vertex in the point, line, or triangle primitive that produced
    the vertex.

    When no tessellation or geometry shader is active, the vertices passed to
    each draw call are arranged into point, line, or triangle primitives as
    described in Section 10.1.  If the <n> vertices passed to a draw call are
    numbered 0 through <n>-1, and the point, line, and triangle primitives
    produced by the draw call are numbered with consecutive integers beginning
    with zero, Table X.1 and Table X.2 indicate the original vertex numbers
    used as vertex 0, vertex 1, and vertex 2 when sourcing per-vertex
    attributes for fragments produced by the primitive numbered <i>.  Table
    X.1 applies when the provoking vertex convention is
    FIRST_VERTEX_CONVENTION, while Table X.2 applies when the provoking vertex
    convention is LAST_VERTEX_CONVENTION.

        Primitive Type                  Vertex 0    Vertex 1    Vertex 2
        ------------------------        --------    --------    --------
        POINTS                          i           -           -
        LINES                           2i          2i+1        -
        LINE_STRIP                      i           i+1         -
        LINE_LOOP                       i           i+1         -
        LINE_LOOP (last segment)        n-1         0           -
        TRIANGLES                       3i          3i+1        3i+2
        TRIANGLE_STRIP (even)           i           i+1         i+2
        TRIANGLE_STRIP (odd)            i           i+2         i+1
        TRIANGLE_FAN                    i+1         i+2         0
        POLYGON                         0           i           i+1
        LINES_ADJACENCY                 4i+1        4i+2        -
        LINES_STRIP_ADJACENCY           i+1         i+2         -
        TRIANGLES_ADJACENCY             6i          6i+2        6i+4
        TRIANGLE_STRIP_ADJACENCY (even) 2i          2i+2        2i+4
        TRIANGLE_STRIP_ADJACENCY (odd)  2i          2i+4        2i+2

        Table X.1, Vertex Order for per-vertex attributes, using the provoking
        vertex convention FIRST_VERTEX_CONVENTION.

        Primitive Type                  Vertex 0    Vertex 1    Vertex 2
        ------------------------        --------    --------    --------
        POINTS                          i           -           -
        LINES                           2i          2i+1        -
        LINE_STRIP                      i           i+1         -
        LINE_LOOP                       i           i+1         -
        LINE_LOOP (last segment)        n-1         0           -
        TRIANGLES                       3i          3i+1        3i+2
        TRIANGLE_STRIP (even)           i           i+1         i+2
        TRIANGLE_STRIP (odd)            i+1         i           i+2
        TRIANGLE_FAN                    0           i+1         i+2
        POLYGON                         0           i           i+1
        LINES_ADJACENCY                 4i+1        4i+2
        LINES_STRIP_ADJACENCY           i+1         i+2
        TRIANGLES_ADJACENCY             6i          6i+2        6i+4
        TRIANGLE_STRIP_ADJACENCY (even) 2i          2i+2        2i+4
        TRIANGLE_STRIP_ADJACENCY (odd)  2i+2        2i          2i+4

        Table X.2, Vertex Order for per-vertex attributes, using the provoking
        vertex convention LAST_VERTEX_CONVENTION.

    When using geometry shaders, vertices used for per-vertex fragment shader
    inputs are determined using Table X.1 or X.2 by treating the primitive(s)
    produced by the geometry shader as though they were passed to a DrawArrays
    calls.  When using a tessellation evaluation shader, or when using QUADS
    or QUAD_STRIP primitives, the vertices used for reading per-vertex
    fragment shader inputs are assigned in an implementation-dependent order.

    The built-in variables gl_BaryCoordNV and gl_BaryCoordNoPerspNV are
    three-component floating-point vectors holding barycentric coordinates for
    the fragment.  These built-ins are computed by clipping (Section 13.6.1)
    and interpolating (Sections 14.5.1 and 14.6.1) a three-component vector
    attribute.  The vertices that are numbered 0, 1, and 2 for the purposes of
    reading per-vertex fragment shader inputs are assigned values of (1,0,0),
    (0,1,0), and (0,0,1), respectively.  For gl_BaryCoordNV, these values are
    clipped and interpolated using perspective correction.  For
    gl_BaryCoordNoPerspNV, these values are clipped and interpolated without
    perspective correction, like other fragment shader inputs qualified with
    "noperspective".


Additions to the AGL/GLX/WGL Specifications

    None

Interactions with OpenGL ES

    Vertex order always corresponds to provoking vertex convention
    LAST_VERTEX_CONVENTION.

    Ignore references to unsupported primitive types QUADS, QUAD_STRIP, and
    POLYGON.

Errors

    None

New State

    None

New Implementation Dependent State

    None

Issues

    (1) Can applications use the original order of vertices in a draw call to
        determine the order of the three vertices used when reading per-vertex
        fragment shader inputs?

      RESOLVED:  Yes, in most cases.

      This extension allows fragment shaders to read inputs qualified with
      "pervertexNV" using a vertex number 0, 1, or 2.  For most primitive
      types, the OpenGL Specification already specifies how the original
      vertices passed to a draw call are assigned to individual point, line,
      or triangle primitives.  The extension extends that language to define a
      specific vertex order that will be used for sourcing per-vertex
      attributes.  In some cases, this vertex order depends on the provoking
      vertex convention.

      When using a tessellation evaluation shader, QUADS primitives, or
      QUAD_STRIP primitives, the OpenGL Specification already indicates that
      patches or quadrilaterals can be decomposed into finer primitives in an
      implementation-dependent order.  In these cases, we do not guarantee a
      specific vertex order.  However, we still guarantee that the vertices
      numbered 0, 1, 2 have corresponding barycentric weights (gl_BaryCoordNV)
      of (1,0,0), (0,1,0), and (0,0,1), respectively.  With this guarantee,
      interpolating attributes manually in a fragment shader with code like:

        float value = (gl_BaryCoordNV.x * v[0].attrib +
                       gl_BaryCoordNV.y * v[1].attrib +
                       gl_BaryCoordNV.z * v[2].attrib);

      should produce results approximately equal to those that would be
      obtained via conventional attribute interpolation.

    (2) How are clipped primitives handled when using "pervertexNV"
        fragment shader inputs?

      RESOLVED:  In the OpenGL pipeline, clipped primitives are normally
      handled by having the clipper remove one of the original vertices,
      introduce one or more new vertices, and process the result as an
      unclipped primitive.  In this model, the provoking vertex still needs to
      be maintained because inputs qualified with "flat" use the values from
      that vertex even if the provoking vertex is clipped.

      In this extension, we guarantee that the three sets of per-vertex values
      available as fragment shader inputs are those of the original primitive
      vertices prior to clipping.  To ensure consistent attribute handling,
      the barycentric weights are computed relative to the original primitive,
      not the clipped one.  For example, if the left half of triangle ABC
      below is clipped away, the clipper introduces a new vertex D and
      rasterizes triangle DBC instead.

                          + B (0,1,0)
                         /|\
                        / | \
                       /  |  \
                      /   |   \
                     /    |    \
                    /     |     \
         (1,0,0) A +------+------+ C (0,0,1)
                          D

      When we process the clipped triangle, the three vertices available for
      "pervertexNV" inputs are actually A, B, and C (in undefined order).
      If vertices "v[0]", "v[1]", and "v[2]" are assigned to A, B, and C,
      respectively, fragments at A, B, and C will have barycentric coordinates
      of (1,0,0), (0,1,0), and (0,0,1), respectively.  A fragment at the
      vertex D introduced by the clipper will have a weight like (0.5, 0.0,
      0.5) -- exactly the same value it would have if ABC were unclipped.

    (3) Should we have any program interface query API support where
        application code can inspect the active fragment shader inputs and
        determine which ones were declared with "pervertexNV"?

      RESOLVED:  No.  We don't have this for other interpolation qualifiers
      like "flat" or "noperspective".

    Also, please refer to issues in the GLSL extension specification.

Revision History

    Revision 2 (mchock)
    - Add support for OpenGL ES.

    Revision 1 (pbrown)
    - Internal revisions.
