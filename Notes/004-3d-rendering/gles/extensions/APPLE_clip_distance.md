# APPLE_clip_distance

Name

    APPLE_clip_distance

Name Strings

    GL_APPLE_clip_distance

Contact

    Eric Sunalp, Apple (esunalp 'at' apple.com)

Contributors

    Keith Bauer
    Alex Eddy
    Benj Lipchak

Status

    Complete

Version

    Last Modified Date: December 9, 2013
    Revision: 1

Number

    OpenGL ES Extension #193

Dependencies

    OpenGL ES 2.0 or OpenGL ES 3.0 is required.

    This extension is written against the OpenGL ES 2.0.25 Specification
    (November 2, 2010) and the OpenGL ES 3.0.2 Specification (April 8, 2013).

Overview

    This extension adds support for hardware clip planes to OpenGL ES 2.0
    and 3.0.  These were present in OpenGL ES 1.1, but were removed to
    better match certain hardware.  Since they're useful for certain
    applications, notable CAD, we return them here.

IP Status

    No known IP claims.

New Tokens

    Accepted by the <pname> parameters of GetBooleanv, GetIntegerv,
    GetInteger64v, and GetFloatv:

        MAX_CLIP_DISTANCES_APPLE       0x0D32

    Accepted by the <pname> parameters of Enable, Disable and IsEnabled:

        CLIP_DISTANCE0_APPLE           0x3000
        CLIP_DISTANCE1_APPLE           0x3001
        CLIP_DISTANCE2_APPLE           0x3002
        CLIP_DISTANCE3_APPLE           0x3003
        CLIP_DISTANCE4_APPLE           0x3004
        CLIP_DISTANCE5_APPLE           0x3005
        CLIP_DISTANCE6_APPLE           0x3006
        CLIP_DISTANCE7_APPLE           0x3007

Additions to Chapter 2 of the OpenGL ES 2.0 Specification (OpenGL Operation)
[ES2] or
Additions to Chapter 2 of the OpenGL ES 3.0 Specification (OpenGL Operation)
[ES3]

Modify Section 2.13, Primitive Clipping (p. 46) [ES2] or
Modify Section 2.17, Primitive Clipping (p. 91) [ES3]

    This view volume may be further restricted by as many as n client-
    defined half-spaces. (n is an implementation-dependent maximum that must
    be at least 8.)

    The clip volume is the intersection of all such half-spaces with the view
    volume (if no client-defined half-spaces are enabled, the clip volume is
    the view volume).

    A vertex shader may write a single clip distance for each supported
    half-space to elements of the gl_ClipDistance[] array. Half-space n is
    then given by the set of points satisfying the inequality

        c_n(P) >= 0,

    where c_n(P) is the value of clip distance n at point P. For point
    primitives, c_n(P) is simply the clip distance for the vertex in question.
    For line and triangle primitives, per-vertex clip distances are
    interpolated using a weighted mean, with weights derived according to the
    algorithms described in sections 3.4 [ES2] or 3.5 [ES3] and
    3.5 [ES2] or 3.6[ES3].

    Client-defined half-spaces are enabled with the generic Enable command
    and disabled with the Disable command. The value of the argument to
    either command is CLIP_DISTANCEi_APPLE, where i is an integer between 0
    and n − 1; specifying a value of i enables or disables the plane equation
    with index i. The constants obey CLIP_DISTANCEi_APPLE =
    CLIP_DISTANCE0_APPLE + i.

Additions to OpenGL ES Shading Language 1.00 Specification [ES2] or
Additions to OpenGL ES Shading Language 3.00 Specification [ES3]

    Including the following line in a shader can be used to control
    the language features described in this extension:

        #extension GL_APPLE_clip_distance : <behavior>

    where <behavior> is as described in section 3.4.

    A new preprocessor #define is added to the OpenGL ES Shading Language:

        #define GL_APPLE_clip_distance 1

Additions to Chapter 7 of the OpenGL ES Shading Language 1.00 Specification 
(Built-in Variables) [ES2] or
Additions to Chapter 7 of the OpenGL ES Shading Language 3.00 Specification
(Built-in Variables) [ES3]

Modify Section 7.1, Vertex Shader Special Variables (p. 59) [ES2]

        varying highp float gl_ClipDistance[];

Modify Section 7.1, Vertex Shader Special Variables (p. 77) [ES3]

        out highp float gl_ClipDistance[];

Modify Section 7.1, Vertex Shader Special Variables (p. 59) [ES2] or
Modify Section 7.1, Vertex Shader Special Variables (p. 77) [ES3]

    The variable gl_ClipDistance provides the mechanism for controlling
    user clipping.  The element gl_ClipDistance[i] specifies a clip distance
    for each plane i.  A distance of 0 means the vertex is on the plane, a
    positive distance means the vertex is inside the clip plane, and a
    negative distance means the point is outside the clip plane.  The clip
    distances will be linearly interpolated across the primitive and the
    portion of the primitive with interpolated distances less than 0 will
    be clipped.

    The gl_ClipDistance array is predeclared as unsized and must be sized by
    the shader either redeclaring it with a size or indexing it only with
    integral constant expressions.  This needs to size the array to include
    all the clip planes that are enabled via the OpenGL ES API; if the size
    does not include all enabled planes, results are undefined.  The size can
    be at most gl_MaxClipDistances.  The number of varying vectors (see
    gl_MaxVaryingVectors) consumed by gl_ClipDistance will match the size
    of the array divided by four, no matter how many planes are enabled.
    The shader must also set all values in gl_ClipDistance that have been
    enabled via the OpenGL API, or results are undefined.  Values written
    into gl_ClipDistance for planes that are not enabled have no effect.

Modify Section 7.4, Built-in Constants (p. 61) [ES2] or
Modify Section 7.3, Built-In Constants (p. 78) [ES3]

    const mediump int gl_MaxClipDistances = 8;

Additions to the AGL/EGL/GLX/WGL Specifications

    None

Errors

    none

New State

                                      Initial
    Get Value      Type   Get Command Value    Description       Sec.
    -------------- ------ ----------- -------- ----------------- ----
    CLIP_DISTANCE- 8* x B IsEnabled   FALSE    ith user clipping 2.13 [ES2] or
    iAPPLE                                     plane enabled     2.17 [ES3]


New Implementation Dependent State

                                      Minimum
    Get Value      Type   Get Command Value    Description       Sec.
    -------------- ------ ----------- -------- ----------------- ----
    MAX_CLIP_DIST- Z+     GetIntegerv 8        Maximum number of 2.13 [ES2] or
    ANCES_APPLE                                user clipping     2.17 [ES3]
                                               planes

Conformance Tests

    Unspecified at this time.

Issues

    (1) GLSL 300 doesn't support unsized arrays, how should gl_ClipDistance
    be sized?

      RESOLVED: For maximal compatibility, it works like ES2/desktop,
      remaining unsized until sized by direct access or explicitly
      redeclared.  Language describing this behavior must be added to the
      extension specification since it's gone from the base language
      specification.

    (2) Should we specify gl_ClipDistance as input to the fragment shader?

      RESOLVED: No.  Although this departs from the desktop, we don't know
      of anyone who needs or wants this behavior, and it complicates the
      driver without adding any additional functionality (the user can
      always pass clip distances to the fragment shader in user-defined
      varyings if they wish).

    (3) Should the GLSL built-in be named gl_ClipDistanceAPPLE?

      RESOLVED: No.  Very few GLSL extensions have adopted name suffixes, it
      would hinder portability of shaders between OpenGL and OpenGL ES, and
      if in the future gl_ClipDistance becomes part of some core ES shading
      language there are no likely incompatibilities with this extension.

Revision History

    Revision 1, 2013/12/13
        - 
