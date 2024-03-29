# OES_fixed_point

Name

    OES_fixed_point

Name Strings

    GL_OES_fixed_point

Contact

    David Blythe (blythe 'at' bluevoid.com)

Notice

    Copyright (c) 2002-2013 The Khronos Group Inc. Copyright terms at
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

    Ratified by the Khronos BOP, July 23, 2003.
    Ratified by the Khronos BOP, Aug 5, 2004.

Version

    Last Modifed Date: 27 April 2015
    Author Revision: 1.0

Number

    OpenGL ES Extension #9 (formerly ARB Extension #292)

Dependencies

    None
    The extension is written against the OpenGL 1.3 Specification.

Overview

    This extension provides the capability, for platforms that do
    not have efficient floating-point support, to input data in a
    fixed-point format, i.e.,  a scaled-integer format.  There are
    several ways a platform could try to solve the problem, such as
    using integer only commands, but there are many OpenGL commands
    that have only floating-point or double-precision floating-point
    parameters.  Also, it is likely that any credible application
    running on such a platform will need to perform some computations
    and will already be using some form of fixed-point representation.
    This extension solves the problem by adding new ``fixed', and
    ``clamp fixed''  data types based on a a two's complement
    S15.16 representation.  New versions of commands are created
    with an 'x' suffix that take fixed or clampx parameters.


IP Status

    None

Issues

*   Add double-precision (S31.32) form too?
      NO

*   Additional InterleavedArray formats?
      NO

*   Should newly suffixed commands, e.g., PointSize, get an alias with
    a float or double suffix for consistency?
      NO

*   Are enums converted to fixed by scaling by 2^16.
      NO.  An enums are passed through as if they are already in
      S15.16 form.  Requiring scaling is too error prone.

New Procedures and Functions

    NOTE:  `T' expands to 'const fixed*' or `fixed' as appropriate

    void Vertex{234}x[v]OES(T coords);
    void Normal3x[v]OES(T coords);
    void TexCoord{1234}x[v]OES(T coords);
    void MultiTexCoord{1234}x[v]OES(enum texture, T coords);
    void Color{34}x[v]OES(T components);
    void Indexx[v]OES(T component);
    void RectxOES(fixed x1, fixed y1, fixed x2, fixed y2);
    void RectxvOES(const fixed v1[2], const fixed v2[2]);

    void DepthRangexOES(clampx n, clampx f);
    void LoadMatrixxOES(const fixed m[16]);
    void MultMatrixxOES(const fixed m[16]);
    void LoadTransposeMatrixxOES(const fixed m[16]);
    void MultTransposeMatrixxOES(const fixed m[16]);
    void RotatexOES(fixed angle, fixed x, fixed y, fixed z);
    void ScalexOES(fixed x, fixed y, fixed z);
    void TranslatexOES(fixed x, fixed y, fixed z);
    void FrustumxOES(fixed l, fixed r, fixed b, fixed t, fixed n, fixed f);
    void OrthoxOES(fixed l, fixed r, fixed b, fixed t, fixed n, fixed f);
    void TexGenx[v]OES(enum coord, enum pname, T param);
    void GetTexGenxvOES(enum coord, enum pname, T* params);

    void ClipPlanexOES(enum plane, const fixed* equation);
    void GetClipPlanexOES(enum plane, fixed* equation);

    void RasterPos{234}x[v]OES(T coords);

    void Materialx[v]OES(enum face, enum pname, T param);
    void GetMaterialxOES(enum face, enum pname, T param);
    void Lightx[v]OES(enum light, enum pname, T* params);
    void GetLightxOES(enum light, enum pname, T* params);
    void LightModelx[v]OES(enum pname, T param);

    void PointSizexOES(fixed size);
    void PointParameterxvOES(enum pname, const fixed *params)
    void LineWidthxOES(fixed width);
    void PolygonOffsetxOES(fixed factor, fixed units);

    void PixelStorex{enum pname, T param);
    void PixelTransferxOES(enum pname, T param);
    void PixelMapx{enum map int size T* values);
    void GetPixelMapxv{enum map int size T* values);

    void ConvolutionParameterx[v]OES(enum target, enum pname, T param);
    void GetConvolutionParameterxvOES(enum target, enum pname, T* params);
    void GetHistogramParameterxvOES(enum target, enum pname, T *params);

    void PixelZoomxOES(fixed xfactor, fixed yfactor);

    void BitmapxOES(sizei width, sizei height, fixed xorig, fixed yorig,
                 fixed xmove, fixed ymove, const ubyte* bitmap);

    void TexParameterx[v]OES(enum target, enum pname, T param);
    void GetTexParameterxvOES(enum target, enum pname, T* params);
    void GetTexLevelParameterxvOES(enum target, int level, enum pname, T* params);
    void PrioritizeTexturesxOES(sizei n, uint* textures, clampx* priorities);
    void TexEnvx[v]OES(enum target, enum pname, T param);
    void GetTexEnvxvOES(enum target, enum pname, T* params);

    void Fogx[v]OES(enum pname, T param);

    void SampleCoveragexOES(clampx value, boolean invert);
    void AlphaFuncxOES(enum func, clampx ref);

    void BlendColorxOES(clampx red, clampx green, clampx blue, clampx alpha);

    void ClearColorxOES(clampx red, clampx green, clampx blue, clampx alpha);
    void ClearDepthxOES(clampx depth);
    void ClearAccumxOES(clampx red, clampx green, clampx blue, clampx alpha);
    void AccumxOES(enum op, fixed value);

    void Map1xOES(enum target, T u1, T u2, int stride, int order, T points);
    void Map2xOES(enum target, T u1, T u2, int ustride, int uorder,
                            T v1, T v2, int vstride, int vorder, T points);
    void MapGrid1xOES(int n, T u1, T u2);
    void MapGrid2xOES(int n, T u1, T u2, T v1, T v2);
    void GetMapxvOES(enum target, enum query, T* v);
    void EvalCoord{12}x[v]OES(T coord);

    void FeedbackBufferxOES(sizei n, enum type, fixed* buffer);
    void PassThroughxOES(fixed token);

    GetFixedvOES(enum pname, fixed* params);


New Tokens

    FIXED_OES                0x140C

Additions to Chapter 2 of the OpenGL 1.3 Specification (OpenGL Operation)

    Section 2.1.1 Floating-Point Computation

      Add the following paragraphs:

      On some platforms, floating-point computations are not sufficiently
      well supported to be used in an OpenGL implementation.  On such
      platforms, fixed-point representations may be a viable substitute for
      floating-point.  Internal computations can use either fixed-point
      or floating-point arithmetic.  Fixed-point computations must be
      accurate to within +/-2^-15.  The maximum representable magnitude
      for a fixed-point number used to represent positional or normal
      coordinates must be at least 2^15; the maximum representable
      magnitude for colors or texture coordinates must be at least 2^10.
      The maximum representable magnitude for all other fixed-point
      values must be at least 2^15.  x*0 = 0*x = 0. 1*x = x*1 = x. x +
      0 = 0 + x = x. 0^0 = 1. Fixed-point computations may lead to
      overflows or underflows.  The results of such computations are
      undefined, but must not lead to GL interruption or termination.


    Section 2.3 GL Command Syntax

      Paragraph 3 is updated to include the 'x' suffix and

      Table 2.1 is modified to include the row:

      ---------------
      | x |  fixed  |
      ---------------

      Table 2.2 is modified to include the rows:

      --------------------------------------------------------------
      | fixed  |  32  | signed 2's complement S15.16 scaled integer|
      --------------------------------------------------------------
      | clampx |  32  | S15.16 scaled integer clamped to [0, 1]    |
      --------------------------------------------------------------

      and the count of the number of rows in the text is changed to 16.

      Add paragraph

      The mapping of GL data types to data types of a specific
      language binding are part of the language binding definition and
      may be platform-dependent.  Type conversion and type promotion
      behavior when mixing actual and formal arguments of different
      data types are specific to the language binding and platform.
      For example, the C language includes automatic conversion
      between integer and floating-point data types, but does not
      include automatic conversion between the int and fixed or
      float and fixed GL types since the fixed data type is not a
      distinct built-in type.  Regardless of language binding,
      the enum type converts to fixed-point without scaling and
      integer types are converted by multiplying by 2^16.



    Section 2.7 Vertex Specification

      Commands are revised to included 'x' suffix.

    Section 2.8 Vertex Arrays

      Table 2.4 Vertex Array Sizes is revised to include the 'fixed' type
      for all commands except EdgeFlagPointer.

      References to Vertex command suffixes are revised to include 'x'.

    Section 2.9 Rectangles

      Revise to include 'x' suffix.

    Section 2.10 Coordinate Transformations

      Revise to include 'x' suffix.  Section 2.10.1 describes clampx.
      Add alternate suffixed versions of Ortho and Frustum.

    Section 2.11 Clipping

      Add alternate suffixed version of ClipPlane.

    Section 2.12 Current Raster Position

      Revise to include 'x' suffix.

    Section 2.13 Colors and Coloring

      Revise to include 'x' suffix and
      Table 2.6 is modified to include row:

      ---------------
      | fixed |  c  |
      ---------------


Additions to Chapter 3 of the OpenGL 1.3 Specification (Rasterization)

    Section 3.3 Points

      Add alternate suffixed PointSize command.

    Section 3.4 Line Segments

      Add alternate suffixed LineWidth command.

    Section 3.5 Polygons

      Add alternate suffixed PolygonOffset command.

    Section 3.6 Pixel Rectangles

      Revise to include 'x' suffix on PixelStore, PixelTransfer, PixelMap,
      ConvolutionParameter.

      Table 3.5 is modified to include row:

      ----------------------
      | FIXED | fixed | No |
      ----------------------

      Add alternate suffixed PixelZoom to Section 3.6.5

    Section 3.7 Bitmaps

      Add alternate suffixed Bitmap command.

    Section 3.8 Texturing

      Revise to include 'x' suffix in TexParameter (Section 3.8.4).

      Add alternate suffixed PrioritizeTextures command (Section 3.8.11).

      Revise to include 'x' suffix in TexEnv (Section 3.8.12).

    Section 3.10 Fog

      Revise to include ;x; suffix in Fog command.


Additions to Chapter 4 of the OpenGL 1.3 Specification (Per-Fragment
Operations and the Frame Buffer)

    Section 4.1 Fragment Operations

      Add alternate suffixed SampleCoveragex (Section 4.1.3), AlphaFunc
      (Section 4.1.4), and BlendColor (Section 4.1.7) commands.

    Section 4.2 Whole Framebuffer Operations

      Add alternate suffixed ClearColor, ClearDepth, and ClearAccum commands
      (Section 4.2.3).

      Add alternate suffixed Accum command (Section 4.2.4).


Additions to Chapter 5 of the OpenGL 1.3 Specification (Special Functions)

    Section 5.1 Evaluators

      Revise to include 'x' suffix on Map1, Map2, Map1Grid, and Map2Grid
      commands.

    Section 5.3 Feedback

      Add alternate suffixed FeedbackBuffer and PassThrough commands.
      Revise Figure 5.2 to indicate 'f' values may also be 'x' values.

Additions to Chapter 6 of the OpenGL 1.3 Specification (State and
State Requests)

      Add GetFixedv to Section 6.1.1.  Revise Section 6.1.2 to
      include implied conversions for GetFixedv.

      Revise to include 'x' suffix for GetClipPlane, GetLightm GetMaterial,
      GetTexEnv, GetTexGen, GetTexParameter, GetTexLevelParameter,
      GetPixelMap, and GetMap in Section 6.1.3.

      Revise to include 'x' suffix for GetHistogramParameter (Section 6.1.9).

    Section 6.2 State Tables

      Revise intro paragraph to include GetFixedv.

Additions to Appendix A of the OpenGL 1.3 Specification (Invariance)

    None

Additions to the AGL/GLX/WGL Specifications

    None

Additions to the WGL Specification

    None

Additions to the AGL Specification

    None

Additions to Chapter 2 of the GLX 1.3 Specification (GLX Operation)

    The data representation is client-side only.  The GLX layer
    performs translation between fixed and float representations.

Additions to Chapter 3 of the GLX 1.3 Specification (Functions and Errors)

Additions to Chapter 4 of the GLX 1.3 Specification (Encoding on the X
Byte Stream)

Additions to Chapter 5 of the GLX 1.3 Specification (Extending OpenGL)

Additions to Chapter 6 of the GLX 1.3 Specification (GLX Versions)

GLX Protocol

    Fixed type entry points are mapped on the client-side to the
    appropriate floating-point command protocol.  To preserve precision,
    double-precision protocol is encouraged, but not required.

Errors

    None

New State

    None

New Implementation Dependent State

    None

Revision History

    12/15/2002    0.1
        - Original draft.

    03/31/2003    0.2
        - Corrected a typo in GetClipPlanex and FIXED_OES.

    04/24/2003    0.3
        - Added clarification that enums must be converted to fixed
          by scaling when passed in a fixed parameter type.  Corrected
          some typos.

    05/29/2003    0.4
        - Changed enums to be passed unscaled when passed to a
          fixed formal parameter.

    07/08/2003    0.5
        - Removed bogus Dependencies on section
        - Added extension number and enumerant value

    07/11/2003    0.6
        - Added OES suffixes

    07/12/2003    0.7
        - Added note about GLX protocol

    06/16/2004    0.8
        - Added ClipPlanex, and various Get functions

    04/27/2015    1.0 (Jon Leech)
        - Replace SampleCoverageOES with SampleCoveragexOES, to match the
          specfile / headers (Bug 13591).
