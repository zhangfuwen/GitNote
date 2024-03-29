# NV_path_rendering

Name

    NV_path_rendering

Name Strings

    GL_NV_path_rendering

Contact

    Mark Kilgard, NVIDIA (mjk 'at' nvidia.com)

Contributors

    Roger Allen, NVIDIA
    Jeff Bolz, NVIDIA
    Chris Dalton, NVIDIA
    Pierre-Loup Griffais, NVIDIA
    Chris Hebert, Samsung
    Scott Nations, NVIDIA
    David Chait, NVIDIA
    Daniel Koch, NVIDIA
    Bas Schouten, Mozilla
    Sandeep Shinde, NVIDIA

Status

    Released in NVIDIA Driver Release 275.33 (June 2011).

    Substantially optimized in NVIDIA Driver Release 301.42 (May 2012).

    Further optimized in NVIDIA Driver Release 314.xx (February 2013).

    Version 1.3 functionality shipping in NVIDIA Driver Release 337.88
    and on (May, 27 2014).

Version

    Last Modified Date:  September 9, 2014
    Version:             35

Number

    OpenGL Extension #410
    OpenGL ES Extension #199

Dependencies

    This extension is written against the OpenGL 3.2 Specification with
    Compatibility Profile but can apply to OpenGL 1.1 and up.

    When used with a Core profile or OpenGL ES context, certain
    functionality is unavailable (see "Dependencies on Core Profile and
    OpenGL ES" section).

    This extension depends on ARB_program_interface_query.

    EXT_direct_state_access commands are used in specifying portions
    of this extension but EXT_direct_state_access is not required to
    implement this extension as long as the functionality implemented
    is equivalent to the EXT_direct_state_access commands.

    EXT_separate_shader_objects is recommended.

    ARB_program_interface_query is recommended.

Overview

    Conventional OpenGL supports rendering images (pixel rectangles and
    bitmaps) and simple geometric primitives (points, lines, polygons).

    This extension adds a new rendering paradigm, known as path rendering,
    for rendering filled and stroked paths.  Path rendering is not novel
    but rather a standard part of most resolution-independent 2D rendering
    systems such as Flash, PDF, Silverlight, SVG, Java 2D, Office
    drawings, TrueType fonts, PostScript and its fonts, Quartz 2D, XML
    Paper Specification (XPS), and OpenVG.  What is novel is the ability
    to mix path rendering with arbitrary OpenGL 3D rendering and imaging.

    With this extension, path rendering becomes a first-class rendering
    mode within the OpenGL graphics system that can be arbitrarily mixed
    with existing OpenGL rendering and can take advantage of OpenGL's
    existing mechanisms for texturing, programmability, and per-fragment
    operations.

    Unlike geometric primitive rendering, paths are specified on a 2D
    (non-projective) plane rather than in 3D (projective) space.
    Even though the path is defined in a 2D plane, every path can
    be transformed into 3D clip space allowing for 3D view frustum &
    user-defined clipping, depth offset, and depth testing in the same
    manner as geometric primitive rendering.

    Both geometric primitive rendering and path rendering support
    rasterization of edges defined by line segments; however, path
    rendering also allows path segments to be specified by Bezier (cubic
    or quadratic) curves or partial elliptical arcs.  This allows path
    rendering to define truly curved primitive boundaries unlike the
    straight edges of line and polygon primitives.  Whereas geometric
    primitive rendering requires convex polygons for well-defined
    rendering results, path rendering allows (and encourages!) concave
    and curved outlines to be specified.  These paths are even allowed
    to self-intersect.

    When filling closed paths, the winding of paths (counterclockwise
    or clockwise) determines whether pixels are inside or outside of
    the path.

    Paths can also be stroked whereby, conceptually, a fixed-width "brush"
    is pulled along the path such that the brush remains orthogonal to
    the gradient of each path segment.  Samples within the sweep of this
    brush are considered inside the stroke of the path.

    This extension supports path rendering through a sequence of three
    operations:

        1.  Path specification is the process of creating and updating
            a path object consisting of a set of path commands and a
            corresponding set of 2D vertices.

            Path commands can be specified explicitly from path command
            and coordinate data, parsed from a string based on standard
            grammars for representing paths, or specified by a particular
            glyph of standard font representations.  Also new paths can
            be specified by weighting one or more existing paths so long
            as all the weighted paths have consistent command sequences.

            Each path object contains zero or more subpaths specified
            by a sequence of line segments, partial elliptical arcs,
            and (cubic or quadratic) Bezier curve segments.  Each path
            may contain multiple subpaths that can be closed (forming
            a contour) or open.

        2.  Path stenciling is the process of updating the stencil buffer
            based on a path's coverage transformed into window space.

            Path stenciling can determine either the filled or stroked
            coverage of a path.

            The details of path stenciling are explained within the core
            of the specification.

            Stenciling a stroked path supports all the standard
            embellishments for path stroking such as end caps, join
            styles, miter limits, dashing, and dash caps.  These stroking
            properties specified are parameters of path objects.

        3.  Path covering is the process of emitting simple (convex &
            planar) geometry that (conservatively) "covers" the path's
            sample coverage in the stencil buffer.  During path covering,
            stencil testing can be configured to discard fragments not
            within the actual coverage of the path as determined by
            prior path stenciling.

            Path covering can cover either the filled or stroked coverage
            of a path.

            The details of path covering are explained within the core
            of the specification.

    To render a path object into the color buffer, an application specifies
    a path object and then uses a two-step rendering process.  First, the
    path object is stenciled whereby the path object's stroked or filled
    coverage is rasterized into the stencil buffer.  Second, the path object
    is covered whereby conservative bounding geometry for the path is
    transformed and rasterized with stencil testing configured to test against
    the coverage information written to the stencil buffer in the first step
    so that only fragments covered by the path are written during this second
    step.  Also during this second step written pixels typically have
    their stencil value reset (so there's no need for clearing the
    stencil buffer between rendering each path).

    Here is an example of specifying and then rendering a five-point
    star and a heart as a path using Scalable Vector Graphics (SVG)
    path description syntax:

        GLuint pathObj = 42;
        const char *svgPathString =
          // star
          "M100,180 L40,10 L190,120 L10,120 L160,10 z"
          // heart
          "M300 300 C 100 400,100 200,300 100,500 200,500 400,300 300Z";
        glPathStringNV(pathObj, GL_PATH_FORMAT_SVG_NV,
                       (GLsizei)strlen(svgPathString), svgPathString);

    Alternatively applications oriented around the PostScript imaging
    model can use the PostScript user path syntax instead:

        const char *psPathString =
          // star
          "100 180 moveto"
          " 40 10 lineto 190 120 lineto 10 120 lineto 160 10 lineto closepath"
          // heart
          " 300 300 moveto"
          " 100 400 100 200 300 100 curveto"
          " 500 200 500 400 300 300 curveto closepath";
        glPathStringNV(pathObj, GL_PATH_FORMAT_PS_NV,
                       (GLsizei)strlen(psPathString), psPathString);

    The PostScript path syntax also supports compact and precise binary
    encoding and includes PostScript-style circular arcs.

    Or the path's command and coordinates can be specified explicitly:

        static const GLubyte pathCommands[10] =
          { GL_MOVE_TO_NV, GL_LINE_TO_NV, GL_LINE_TO_NV, GL_LINE_TO_NV,
            GL_LINE_TO_NV, GL_CLOSE_PATH_NV,
            'M', 'C', 'C', 'Z' };  // character aliases
        static const GLshort pathCoords[12][2] =
          { {100, 180}, {40, 10}, {190, 120}, {10, 120}, {160, 10},
            {300,300}, {100,400}, {100,200}, {300,100},
            {500,200}, {500,400}, {300,300} };
        glPathCommandsNV(pathObj, 10, pathCommands, 24, GL_SHORT, pathCoords);

    Before rendering to a window with a stencil buffer, clear the stencil
    buffer to zero and the color buffer to black:

        glClearStencil(0);
        glClearColor(0,0,0,0);
        glStencilMask(~0);
        glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    Use an orthographic path-to-clip-space transform to map the
    [0..500]x[0..400] range of the star's path coordinates to the [-1..1]
    clip space cube:

        glMatrixLoadIdentityEXT(GL_PROJECTION);
        glMatrixLoadIdentityEXT(GL_MODELVIEW);
        glMatrixOrthoEXT(GL_MODELVIEW, 0, 500, 0, 400, -1, 1);

    Stencil the path:

        glStencilFillPathNV(pathObj, GL_COUNT_UP_NV, 0x1F);

    The 0x1F mask means the counting uses modulo-32 arithmetic. In
    principle the star's path is simple enough (having a maximum winding
    number of 2) that modulo-4 arithmetic would be sufficient so the mask
    could be 0x3.  Or a mask of all 1's (~0) could be used to count with
    all available stencil bits.

    Now that the coverage of the star and the heart have been rasterized
    into the stencil buffer, cover the path with a non-zero fill style
    (indicated by the GL_NOTEQUAL stencil function with a zero reference
    value):

        glEnable(GL_STENCIL_TEST);
        glStencilFunc(GL_NOTEQUAL, 0, 0x1F);
        glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);
        glColor3f(1,1,0); // yellow
        glCoverFillPathNV(pathObj, GL_BOUNDING_BOX_NV);

    The result is a yellow star (with a filled center) to the left of
    a yellow heart.

    The GL_ZERO stencil operation ensures that any covered samples
    (meaning those with non-zero stencil values) are zero'ed when
    the path cover is rasterized. This allows subsequent paths to be
    rendered without clearing the stencil buffer again.

    A similar two-step rendering process can draw a white outline
    over the star and heart.

    Before rendering, configure the path object with desirable path
    parameters for stroking.  Specify a wider 6.5-unit stroke and
    the round join style:

        glPathParameteriNV(pathObj, GL_PATH_JOIN_STYLE_NV, GL_ROUND_NV);
        glPathParameterfNV(pathObj, GL_PATH_STROKE_WIDTH_NV, 6.5);

     Now stencil the path's stroked coverage into the stencil buffer,
     setting the stencil to 0x1 for all stencil samples within the
     transformed path.

        glStencilStrokePathNV(pathObj, 0x1, ~0);

     Cover the path's stroked coverage (with a hull this time instead
     of a bounding box; the choice doesn't really matter here) while
     stencil testing that writes white to the color buffer and again
     zero the stencil buffer.

        glColor3f(1,1,1); // white
        glCoverStrokePathNV(pathObj, GL_CONVEX_HULL_NV);

     In this example, constant color shading is used but the application
     can specify their own arbitrary shading and/or blending operations,
     whether with Cg compiled to fragment program assembly, GLSL, or
     fixed-function fragment processing.

     More complex path rendering is possible such as clipping one path to
     another arbitrary path.  This is because stencil testing (as well
     as depth testing, depth bound test, clip planes, and scissoring)
     can restrict path stenciling.

     Now let's render the word "OpenGL" atop the star and heart.

     First create a sequence of path objects for the glyphs for the
     characters in "OpenGL":

        GLuint glyphBase = glGenPathsNV(6);
        const unsigned char *word = "OpenGL";
        const GLsizei wordLen = (GLsizei)strlen(word);
        const GLfloat emScale = 2048;  // match TrueType convention
        GLuint templatePathObject = ~0;  // Non-existent path object
        glPathGlyphsNV(glyphBase,
                       GL_SYSTEM_FONT_NAME_NV, "Helvetica", GL_BOLD_BIT_NV,
                       wordLen, GL_UNSIGNED_BYTE, word,
                       GL_SKIP_MISSING_GLYPH_NV, ~0, emScale);
        glPathGlyphsNV(glyphBase,
                       GL_SYSTEM_FONT_NAME_NV, "Arial", GL_BOLD_BIT_NV,
                       wordLen, GL_UNSIGNED_BYTE, word,
                       GL_SKIP_MISSING_GLYPH_NV, ~0, emScale);
        glPathGlyphsNV(glyphBase,
                       GL_STANDARD_FONT_NAME_NV, "Sans", GL_BOLD_BIT_NV,
                       wordLen, GL_UNSIGNED_BYTE, word,
                       GL_USE_MISSING_GLYPH_NV, ~0, emScale);

    Glyphs are loaded for three different fonts in priority order:
    Helvetica first, then Arial, and if neither of those loads, use the
    standard sans-serif font.  If a prior glPathGlyphsNV is successful
    and specifies the path object range, the subsequent glPathGlyphsNV
    commands silently avoid re-specifying the already existent path
    objects.

    Now query the (kerned) separations for the word "OpenGL" and build
    a set of horizontal translations advancing each successive glyph by
    its kerning distance with the following glyph.

        GLfloat xtranslate[6+1];  // wordLen+1
        glGetPathSpacingNV(GL_ACCUM_ADJACENT_PAIRS_NV,
                           wordLen+1, GL_UNSIGNED_BYTE,
                           "\000\001\002\003\004\005\005",  // repeat last letter twice
                           glyphBase,
                           1.0f, 1.0f,
                           GL_TRANSLATE_X_NV,
                           xtranslate);

    Next determine the font-wide vertical minimum and maximum for the
    font face by querying the per-font metrics of any one of the glyphs
    from the font face.

        GLfloat yMinMax[2];
        glGetPathMetricRangeNV(GL_FONT_Y_MIN_BOUNDS_BIT_NV|GL_FONT_Y_MAX_BOUNDS_BIT_NV,
                               glyphBase, /*count*/1,
                               2*sizeof(GLfloat),
                               yMinMax);

    Use an orthographic path-to-clip-space transform to map the
    word's bounds to the [-1..1] clip space cube:

        glMatrixLoadIdentityEXT(GL_PROJECTION);
        glMatrixOrthoEXT(GL_MODELVIEW,
                         0, xtranslate[6], yMinMax[0], yMinMax[1],
                         -1, 1);

    Stencil the filled paths of the sequence of glyphs for "OpenGL",
    each transformed by the appropriate 2D translations for spacing.

        glStencilFillPathInstancedNV(6, GL_UNSIGNED_BYTE,
                                     "\000\001\002\003\004\005",
                                     glyphBase,
                                     GL_PATH_FILL_MODE_NV, 0xFF,
                                     GL_TRANSLATE_X_NV, xtranslate);

     Cover the bounding box union of the glyphs with 50% gray.

        glEnable(GL_STENCIL_TEST);
        glStencilFunc(GL_NOTEQUAL, 0, 0xFF);
        glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);
        glColor3f(0.5,0.5,0.5); // 50% gray
        glCoverFillPathInstancedNV(6, GL_UNSIGNED_BYTE,
                                   "\000\001\002\003\004\005",
                                   glyphBase,
                                   GL_BOUNDING_BOX_OF_BOUNDING_BOXES_NV,
                                   GL_TRANSLATE_X_NV, xtranslate);

    Voila, the word "OpenGL" in gray is now stenciled into the framebuffer.

    Instead of solid 50% gray, the cover operation can apply a linear
    gradient that changes from green (RGB=0,1,0) at the top of the word
    "OpenGL" to blue (RGB=0,0,1) at the bottom of "OpenGL":

        GLfloat rgbGen[3][3] = {
          0, 0, 0,  // red   = constant zero
          0, 1, 0,  // green = varies with y from bottom (0) to top (1)
          0, -1, 1  // blue  = varies with y from bottom (1) to top (0)
        };
        glPathColorGenNV(GL_PRIMARY_COLOR, GL_PATH_OBJECT_BOUNDING_BOX_NV,
                         GL_RGB, &rgbGen[0][0]);

    Instead of loading just the glyphs for the characters in "OpenGL",
    the entire character set could be loaded.  This allows the characters
    of the string to be mapped (offset by the glyphBase) to path object names.
    A range of glyphs can be loaded like this:

        const int numChars = 256;  // ISO/IEC 8859-1 8-bit character range
        GLuint glyphBase = glGenPathsNV(numChars);
        glPathGlyphRangeNV(glyphBase,
                           GL_SYSTEM_FONT_NAME_NV, "Helvetica", GL_BOLD_BIT_NV,
                           0, numChars,
                           GL_SKIP_MISSING_GLYPH_NV, ~0, emScale);
        glPathGlyphRangeNV(glyphBase,
                           GL_SYSTEM_FONT_NAME_NV, "Arial", GL_BOLD_BIT_NV,
                           0, numChars,
                           GL_SKIP_MISSING_GLYPH_NV, ~0, emScale);
        glPathGlyphRangeNV(glyphBase,
                           GL_STANDARD_FONT_NAME_NV, "Sans", GL_BOLD_BIT_NV,
                           0, numChars,
                           GL_USE_MISSING_GLYPH_NV, ~0, emScale);

    Given a range of glyphs loaded as path objects, (kerned) spacing
    information can now be queried for the string:

        glGetPathSpacingNV(GL_ACCUM_ADJACENT_PAIRS_NV,
                           7, GL_UNSIGNED_BYTE, "OpenGLL", // repeat L to get final spacing
                           glyphBase,
                           1.0f, 1.0f,
                           GL_TRANSLATE_X_NV,
                           kerning);

    Using the range of glyphs, stenciling and covering the instanced
    paths for "OpenGL" can be done this way:

        glStencilFillPathInstancedNV(6, GL_UNSIGNED_BYTE, "OpenGL",
                                     glyphBase,
                                     GL_PATH_FILL_MODE_NV, 0xFF,
                                     GL_TRANSLATE_X_NV, xtranslate);

        glCoverFillPathInstancedNV(6, GL_UNSIGNED_BYTE, "OpenGL",
                                   glyphBase,
                                   GL_BOUNDING_BOX_OF_BOUNDING_BOXES_NV,
                                   GL_TRANSLATE_X_NV, xtranslate);

    The "stencil" and "cover" steps can be combined in a single command:

        glStencilThenCoverFillPathInstancedNV(6, GL_UNSIGNED_BYTE, "OpenGL",
                                              glyphBase,
                                              GL_PATH_FILL_MODE_NV, 0xFF,
                                              GL_BOUNDING_BOX_OF_BOUNDING_BOXES_NV
                                              GL_TRANSLATE_X_NV, xtranslate);

    XXX add path clipping example to demonstrate glPathStencilFuncNV.

New Procedures and Functions

    PATH SPECIFICATION COMMANDS

        EXPLICIT PATH DATA

        void PathCommandsNV(uint path,
                            sizei numCommands, const ubyte *commands,
                            sizei numCoords, enum coordType,
                            const void *coords);
        void PathCoordsNV(uint path,
                          sizei numCoords, enum coordType,
                          const void *coords);

        void PathSubCommandsNV(uint path,
                               sizei commandStart, sizei commandsToDelete,
                               sizei numCommands, const ubyte *commands,
                               sizei numCoords, enum coordType,
                               const void *coords);
        void PathSubCoordsNV(uint path,
                             sizei coordStart,
                             sizei numCoords, enum coordType,
                             const void *coords);

        STRING PATH DESCRIPTION

        void PathStringNV(uint path, enum format,
                          sizei length, const void *pathString);

        PATHS FROM FONT GLYPHS BY UNICODE CHARACTER POINT

        void PathGlyphsNV(uint firstPathName,
                          enum fontTarget,
                          const void *fontName,
                          bitfield fontStyle,
                          sizei numGlyphs, enum type,
                          const void *charcodes,
                          enum handleMissingGlyphs,
                          uint pathParameterTemplate,
                          float emScale);
        void PathGlyphRangeNV(uint firstPathName,
                              enum fontTarget,
                              const void *fontName,
                              bitfield fontStyle,
                              uint firstGlyph,
                              sizei numGlyphs,
                              enum handleMissingGlyphs,
                              uint pathParameterTemplate,
                              float emScale);

        PATHS FROM FONT GLYPHS BY PER-FONT GLYPH INDEX

        enum PathGlyphIndexArrayNV(uint firstPathName,
                                   enum fontTarget,
                                   const void *fontName,
                                   bitfield fontStyle,
                                   uint firstGlyphIndex,
                                   sizei numGlyphs,
                                   uint pathParameterTemplate,
                                   float emScale);
        enum PathMemoryGlyphIndexArrayNV(uint firstPathName,
                                         enum fontTarget,
                                         sizeiptr fontSize,
                                         const void *fontData,
                                         sizei faceIndex,
                                         uint firstGlyphIndex,
                                         sizei numGlyphs,
                                         uint pathParameterTemplate,
                                         float emScale);
        enum PathGlyphIndexRangeNV(enum fontTarget,
                                   const void *fontName,
                                   bitfield fontStyle,
                                   uint pathParameterTemplate,
                                   float emScale,
                                   uint* baseAndCount);

        PATH SPECIFICATION WITH EXISTING PATHS

        void WeightPathsNV(uint resultPath,
                           sizei numPaths,
                           const uint paths[], const float weights[]);
        void CopyPathNV(uint resultPath, uint srcPath);
        void InterpolatePathsNV(uint resultPath,
                                uint pathA, uint pathB,
                                float weight);
        void TransformPathNV(uint resultPath,
                             uint srcPath,
                             enum transformType,
                             const float *transformValues);

    PATH PARAMETER SPECIFICATION COMMANDS

        void PathParameterivNV(uint path, enum pname, const int *value);
        void PathParameteriNV(uint path, enum pname, int value);
        void PathParameterfvNV(uint path, enum pname, const float *value);
        void PathParameterfNV(uint path, enum pname, float value);

        void PathDashArrayNV(uint path,
                             sizei dashCount, const float *dashArray);

    PATH NAME MANAGEMENT

        uint GenPathsNV(sizei range);
        void DeletePathsNV(uint path, sizei range);
        boolean IsPathNV(uint path);

    PATH STENCILING

        void PathStencilFuncNV(enum func, int ref, uint mask);
        void PathStencilDepthOffsetNV(float factor, float units);

        void StencilFillPathNV(uint path,
                               enum fillMode, uint mask);

        void StencilStrokePathNV(uint path,
                                 int reference, uint mask);

        void StencilFillPathInstancedNV(sizei numPaths,
                                        enum pathNameType, const void *paths,
                                        uint pathBase,
                                        enum fillMode, uint mask,
                                        enum transformType,
                                        const float *transformValues);

        void StencilStrokePathInstancedNV(sizei numPaths,
                                          enum pathNameType, const void *paths,
                                          uint pathBase,
                                          int reference, uint mask,
                                          enum transformType,
                                          const float *transformValues);

    PATH COVERING

        void PathCoverDepthFuncNV(enum zfunc);

        void PathColorGenNV(enum color,
                            enum genMode,
                            enum colorFormat, const float *coeffs);
        void PathTexGenNV(enum texCoordSet,
                          enum genMode,
                          int components, const float *coeffs);
        void PathFogGenNV(enum genMode);

        void CoverFillPathNV(uint path, enum coverMode);

        void CoverStrokePathNV(uint path, enum coverMode);

        void CoverFillPathInstancedNV(sizei numPaths,
                                      enum pathNameType, const void *paths,
                                      uint pathBase,
                                      enum coverMode,
                                      enum transformType,
                                      const float *transformValues);

        void CoverStrokePathInstancedNV(sizei numPaths,
                                        enum pathNameType, const void *paths,
                                        uint pathBase,
                                        enum coverMode,
                                        enum transformType,
                                        const float *transformValues);

    PATH STENCILING THEN COVERING

        void StencilThenCoverFillPathNV(uint path, enum fillMode,
                                        uint mask, enum coverMode);
        void StencilThenCoverStrokePathNV(uint path, int reference,
                                          uint mask, enum coverMode);
        void StencilThenCoverFillPathInstancedNV(sizei numPaths,
                                                 enum pathNameType,
                                                 const void *paths,
                                                 uint pathBase,
                                                 enum fillMode, uint mask,
                                                 enum coverMode,
                                                 enum transformType,
                                                 const float *transformValues);
        void StencilThenCoverStrokePathInstancedNV(sizei numPaths,
                                                   enum pathNameType,
                                                   const void *paths,
                                                   uint pathBase,
                                                   int reference, uint mask,
                                                   enum coverMode,
                                                   enum transformType,
                                                   const float *transformValues);

    PATH COVERING OF GLSL FRAGMENT INPUTS

        void ProgramPathFragmentInputGenNV(uint program,
                                           int location,
                                           enum genMode,
                                           int components,
                                           const float *coeffs);

    PATH QUERIES

        void GetPathParameterivNV(uint path, enum pname, int *value);
        void GetPathParameterfvNV(uint path, enum pname, float *value);

        void GetPathCommandsNV(uint path, ubyte *commands);
        void GetPathCoordsNV(uint path, float *coords);
        void GetPathDashArrayNV(uint path, float *dashArray);

        void GetPathMetricsNV(bitfield metricQueryMask,
                              sizei numPaths,
                              enum pathNameType, const void *paths,
                              uint pathBase,
                              sizei stride,
                              float *metrics);
        void GetPathMetricRangeNV(bitfield metricQueryMask,
                                  uint firstPathName,
                                  sizei numPaths,
                                  sizei stride,
                                  float *metrics);

        void GetPathSpacingNV(enum pathListMode,
                              sizei numPaths,
                              enum pathNameType, const void *paths,
                              uint pathBase,
                              float advanceScale,
                              float kerningScale,
                              enum transformType,
                              float *returnedSpacing);

        void GetPathColorGenivNV(enum color, enum pname, int *value);
        void GetPathColorGenfvNV(enum color, enum pname, float *value);
        void GetPathTexGenivNV(enum texCoordSet, enum pname, int *value);
        void GetPathTexGenfvNV(enum texCoordSet, enum pname, float *value);

        boolean IsPointInFillPathNV(uint path,
                                    uint mask, float x, float y);
        boolean IsPointInStrokePathNV(uint path,
                                      float x, float y);

        float GetPathLengthNV(uint path,
                              sizei startSegment, sizei numSegments);

        boolean PointAlongPathNV(uint path,
                                 sizei startSegment, sizei numSegments,
                                 float distance,
                                 float *x, float *y,
                                 float *tangentX, float *tangentY);

    MATRIX SPECIFICATION

        void MatrixLoad3x2fNV(enum matrixMode, const float *m);
        void MatrixLoad3x3fNV(enum matrixMode, const float *m);
        void MatrixLoadTranspose3x3fNV(enum matrixMode, const float *m);

        void MatrixMult3x2fNV(enum matrixMode, const float *m);
        void MatrixMult3x3fNV(enum matrixMode, const float *m);
        void MatrixMultTranspose3x3fNV(enum matrixMode, const float *m);

    FLOATING-POINT PROGRAM RESOURCE QUERY

        void GetProgramResourcefvNV(uint program, enum programInterface,
                                    uint index, sizei propCount,
                                    const enum *props, sizei bufSize,
                                    sizei *length, float *params);

New Tokens

    Accepted in elements of the <commands> array parameter of
    PathCommandsNV and PathSubCommandsNV:

        CLOSE_PATH_NV                                   0x00
        MOVE_TO_NV                                      0x02
        RELATIVE_MOVE_TO_NV                             0x03
        LINE_TO_NV                                      0x04
        RELATIVE_LINE_TO_NV                             0x05
        HORIZONTAL_LINE_TO_NV                           0x06
        RELATIVE_HORIZONTAL_LINE_TO_NV                  0x07
        VERTICAL_LINE_TO_NV                             0x08
        RELATIVE_VERTICAL_LINE_TO_NV                    0x09
        QUADRATIC_CURVE_TO_NV                           0x0A
        RELATIVE_QUADRATIC_CURVE_TO_NV                  0x0B
        CUBIC_CURVE_TO_NV                               0x0C
        RELATIVE_CUBIC_CURVE_TO_NV                      0x0D
        SMOOTH_QUADRATIC_CURVE_TO_NV                    0x0E
        RELATIVE_SMOOTH_QUADRATIC_CURVE_TO_NV           0x0F
        SMOOTH_CUBIC_CURVE_TO_NV                        0x10
        RELATIVE_SMOOTH_CUBIC_CURVE_TO_NV               0x11
        SMALL_CCW_ARC_TO_NV                             0x12
        RELATIVE_SMALL_CCW_ARC_TO_NV                    0x13
        SMALL_CW_ARC_TO_NV                              0x14
        RELATIVE_SMALL_CW_ARC_TO_NV                     0x15
        LARGE_CCW_ARC_TO_NV                             0x16
        RELATIVE_LARGE_CCW_ARC_TO_NV                    0x17
        LARGE_CW_ARC_TO_NV                              0x18
        RELATIVE_LARGE_CW_ARC_TO_NV                     0x19
        CONIC_CURVE_TO_NV                               0x1A
        RELATIVE_CONIC_CURVE_TO_NV                      0x1B
        ROUNDED_RECT_NV                                 0xE8
        RELATIVE_ROUNDED_RECT_NV                        0xE9
        ROUNDED_RECT2_NV                                0xEA
        RELATIVE_ROUNDED_RECT2_NV                       0xEB
        ROUNDED_RECT4_NV                                0xEC
        RELATIVE_ROUNDED_RECT4_NV                       0xED
        ROUNDED_RECT8_NV                                0xEE
        RELATIVE_ROUNDED_RECT8_NV                       0xEF
        RESTART_PATH_NV                                 0xF0
        DUP_FIRST_CUBIC_CURVE_TO_NV                     0xF2
        DUP_LAST_CUBIC_CURVE_TO_NV                      0xF4
        RECT_NV                                         0xF6
        RELATIVE_RECT_NV                                0xF7
        CIRCULAR_CCW_ARC_TO_NV                          0xF8
        CIRCULAR_CW_ARC_TO_NV                           0xFA
        CIRCULAR_TANGENT_ARC_TO_NV                      0xFC
        ARC_TO_NV                                       0xFE
        RELATIVE_ARC_TO_NV                              0xFF

    Accepted by the <format> parameter of PathStringNV:

        PATH_FORMAT_SVG_NV                              0x9070
        PATH_FORMAT_PS_NV                               0x9071

    Accepted by the <fontTarget> parameter of PathGlyphsNV,
    PathGlyphRangeNV, and PathGlyphIndexRangeNV:

        STANDARD_FONT_NAME_NV                           0x9072
        SYSTEM_FONT_NAME_NV                             0x9073
        FILE_NAME_NV                                    0x9074

    Accepted by the <fontTarget> parameter of PathMemoryGlyphIndexArrayNV:

        STANDARD_FONT_FORMAT_NV                         0x936C

    Accepted by the <handleMissingGlyph> parameter of PathGlyphsNV and
    PathGlyphRangeNV:

        SKIP_MISSING_GLYPH_NV                           0x90A9
        USE_MISSING_GLYPH_NV                            0x90AA

    Returned by PathGlyphIndexRangeNV:

        FONT_GLYPHS_AVAILABLE_NV                        0x9368
        FONT_TARGET_UNAVAILABLE_NV                      0x9369
        FONT_UNAVAILABLE_NV                             0x936A
        FONT_UNINTELLIGIBLE_NV                          0x936B  // once was FONT_CORRUPT_NV
        INVALID_ENUM
        INVALID_VALUE
        OUT_OF_MEMORY

    Accepted by the <pname> parameter of PathParameterfNV,
    PathParameterfvNV, GetPathParameterfvNV, PathParameteriNV,
    PathParameterivNV, and GetPathParameterivNV:

        PATH_STROKE_WIDTH_NV                            0x9075
        PATH_INITIAL_END_CAP_NV                         0x9077
        PATH_TERMINAL_END_CAP_NV                        0x9078
        PATH_JOIN_STYLE_NV                              0x9079
        PATH_MITER_LIMIT_NV                             0x907A
        PATH_INITIAL_DASH_CAP_NV                        0x907C
        PATH_TERMINAL_DASH_CAP_NV                       0x907D
        PATH_DASH_OFFSET_NV                             0x907E
        PATH_CLIENT_LENGTH_NV                           0x907F
        PATH_DASH_OFFSET_RESET_NV                       0x90B4

        PATH_FILL_MODE_NV                               0x9080
        PATH_FILL_MASK_NV                               0x9081
        PATH_FILL_COVER_MODE_NV                         0x9082
        PATH_STROKE_COVER_MODE_NV                       0x9083
        PATH_STROKE_MASK_NV                             0x9084
        PATH_STROKE_BOUND_NV                            0x9086

    Accepted by the <pname> parameter of PathParameterfNV and
    PathParameterfvNV:

        PATH_END_CAPS_NV                                0x9076
        PATH_DASH_CAPS_NV                               0x907B

    Accepted by the <fillMode> parameter of StencilFillPathNV and
    StencilFillPathInstancedNV:

        INVERT
        COUNT_UP_NV                                     0x9088
        COUNT_DOWN_NV                                   0x9089
        PATH_FILL_MODE_NV                               see above

    Accepted by the <color> parameter of PathColorGenNV,
    GetPathColorGenivNV, and GetPathColorGenfvNV:

        PRIMARY_COLOR                                   0x8577  // from OpenGL 1.3
        PRIMARY_COLOR_NV                                0x852C  // from NV_register_combiners
        SECONDARY_COLOR_NV                              0x852D  // from NV_register_combiners

    Accepted by the <genMode> parameter of PathColorGenNV, PathTexGenNV,
    ProgramPathFragmentInputGenNV:

        NONE
        EYE_LINEAR
        OBJECT_LINEAR
        PATH_OBJECT_BOUNDING_BOX_NV                     0x908A
        CONSTANT

    Accepted by the <coverMode> parameter of CoverFillPathNV and
    CoverFillPathInstancedNV:

        CONVEX_HULL_NV                                  0x908B
        BOUNDING_BOX_NV                                 0x908D
        PATH_FILL_COVER_MODE_NV                         see above

    Accepted by the <coverMode> parameter of CoverStrokePathNV and
    CoverStrokePathInstancedNV:

        CONVEX_HULL_NV                                  see above
        BOUNDING_BOX_NV                                 see above
        PATH_STROKE_COVER_MODE_NV                       see above

    Accepted by the <transformType> parameter of
    StencilFillPathInstancedNV, StencilStrokePathInstancedNV,
    CoverFillPathInstancedNV, and CoverStrokePathInstancedNV:

        NONE
        TRANSLATE_X_NV                                  0x908E
        TRANSLATE_Y_NV                                  0x908F
        TRANSLATE_2D_NV                                 0x9090
        TRANSLATE_3D_NV                                 0x9091
        AFFINE_2D_NV                                    0x9092
        AFFINE_3D_NV                                    0x9094
        TRANSPOSE_AFFINE_2D_NV                          0x9096
        TRANSPOSE_AFFINE_3D_NV                          0x9098

    Accepted by the <transformType> parameter of TransformPathNV:

        NONE
        TRANSLATE_X_NV                                  see above
        TRANSLATE_Y_NV                                  see above
        TRANSLATE_2D_NV                                 see above
        TRANSLATE_3D_NV                                 see above
        AFFINE_2D_NV                                    see above
        AFFINE_3D_NV                                    see above
        TRANSPOSE_AFFINE_2D_NV                          see above
        TRANSPOSE_AFFINE_3D_NV                          see above

    Accepted by the <type> or <pathNameType> parameter of
    StencilFillPathInstancedNV, StencilStrokePathInstancedNV,
    CoverFillPathInstancedNV, CoverStrokePathInstancedNV,
    GetPathMetricsNV, and GetPathSpacingNV:

        UTF8_NV                                         0x909A
        UTF16_NV                                        0x909B

    Accepted by the <coverMode> parameter of CoverFillPathInstancedNV:

        CONVEX_HULL_NV                                  see above
        BOUNDING_BOX_NV                                 see above
        BOUNDING_BOX_OF_BOUNDING_BOXES_NV               0x909C
        PATH_FILL_COVER_MODE_NV                         see above

    Accepted by the <coverMode> parameter of CoverStrokePathInstancedNV:

        CONVEX_HULL_NV                                  see above
        BOUNDING_BOX_NV                                 see above
        BOUNDING_BOX_OF_BOUNDING_BOXES_NV               see above
        PATH_STROKE_COVER_MODE_NV                       see above

    Accepted by the <pname> parameter of GetPathParameterfvNV and
    GetPathParameterivNV:

        PATH_COMMAND_COUNT_NV                           0x909D
        PATH_COORD_COUNT_NV                             0x909E
        PATH_DASH_ARRAY_COUNT_NV                        0x909F

        PATH_COMPUTED_LENGTH_NV                         0x90A0

        PATH_OBJECT_BOUNDING_BOX_NV                     see above
        PATH_FILL_BOUNDING_BOX_NV                       0x90A1
        PATH_STROKE_BOUNDING_BOX_NV                     0x90A2

    Accepted by the <value> parameter of PathParameterfNV,
    PathParameterfvNV, PathParameteriNV, and PathParameterivNV
    when <pname> is one of PATH_END_CAPS_NV, PATH_INTIAL_END_CAP_NV,
    PATH_TERMINAL_END_CAP_NV, PATH_DASH_CAPS_NV, PATH_INITIAL_DASH_CAP_NV,
    and PATH_TERMINAL_DASH_CAP_NV:

        FLAT
        SQUARE_NV                                       0x90A3
        ROUND_NV                                        0x90A4
        TRIANGULAR_NV                                   0x90A5

    Accepted by the <value> parameter of PathParameterfNV,
    PathParameterfvNV, PathParameteriNV, and PathParameterivNV
    when <pname> is PATH_JOIN_STYLE_NV:

        NONE
        ROUND_NV                                        see above
        BEVEL_NV                                        0x90A6
        MITER_REVERT_NV                                 0x90A7
        MITER_TRUNCATE_NV                               0x90A8

    Accepted by the <value> parameter of PathParameterfNV,
    PathParameterfvNV, PathParameteriNV, and PathParameterivNV when
    <pname> is PATH_DASH_OFFSET_RESET_NV:

        MOVE_TO_RESETS_NV                               0x90B5
        MOVE_TO_CONTINUES_NV                            0x90B6

    Accepted by the <fontStyle> parameter of PathGlyphsNV,
    PathGlyphRangeNV, and PathGlyphIndexRangeNV:

        NONE
        BOLD_BIT_NV                                     0x01
        ITALIC_BIT_NV                                   0x02

    Accepted by the <pname> parameter of GetBooleanv, GetIntegerv,
    GetInteger64v, GetFloatv, and GetDoublev:

        PATH_ERROR_POSITION_NV                          0x90AB

        PATH_FOG_GEN_MODE_NV                            0x90AC

        PATH_STENCIL_FUNC_NV                            0x90B7
        PATH_STENCIL_REF_NV                             0x90B8
        PATH_STENCIL_VALUE_MASK_NV                      0x90B9

        PATH_STENCIL_DEPTH_OFFSET_FACTOR_NV             0x90BD
        PATH_STENCIL_DEPTH_OFFSET_UNITS_NV              0x90BE

        PATH_COVER_DEPTH_FUNC_NV                        0x90BF

    Accepted as a bit within the <metricQueryMask> parameter of
    GetPathMetricRangeNV or GetPathMetricsNV:

        // per-glyph metrics
        GLYPH_WIDTH_BIT_NV                              0x01
        GLYPH_HEIGHT_BIT_NV                             0x02
        GLYPH_HORIZONTAL_BEARING_X_BIT_NV               0x04
        GLYPH_HORIZONTAL_BEARING_Y_BIT_NV               0x08
        GLYPH_HORIZONTAL_BEARING_ADVANCE_BIT_NV         0x10
        GLYPH_VERTICAL_BEARING_X_BIT_NV                 0x20
        GLYPH_VERTICAL_BEARING_Y_BIT_NV                 0x40
        GLYPH_VERTICAL_BEARING_ADVANCE_BIT_NV           0x80
        GLYPH_HAS_KERNING_BIT_NV                        0x100

        // per-font face metrics
        FONT_X_MIN_BOUNDS_BIT_NV                        0x00010000
        FONT_Y_MIN_BOUNDS_BIT_NV                        0x00020000
        FONT_X_MAX_BOUNDS_BIT_NV                        0x00040000
        FONT_Y_MAX_BOUNDS_BIT_NV                        0x00080000
        FONT_UNITS_PER_EM_BIT_NV                        0x00100000
        FONT_ASCENDER_BIT_NV                            0x00200000
        FONT_DESCENDER_BIT_NV                           0x00400000
        FONT_HEIGHT_BIT_NV                              0x00800000
        FONT_MAX_ADVANCE_WIDTH_BIT_NV                   0x01000000
        FONT_MAX_ADVANCE_HEIGHT_BIT_NV                  0x02000000
        FONT_UNDERLINE_POSITION_BIT_NV                  0x04000000
        FONT_UNDERLINE_THICKNESS_BIT_NV                 0x08000000
        FONT_HAS_KERNING_BIT_NV                         0x10000000
        FONT_NUM_GLYPH_INDICES_BIT_NV                   0x20000000

    Accepted by the <pathListMode> parameter of GetPathSpacingNV:

        ACCUM_ADJACENT_PAIRS_NV                         0x90AD
        ADJACENT_PAIRS_NV                               0x90AE
        FIRST_TO_REST_NV                                0x90AF

    Accepted by the <pname> parameter of GetPathColorGenivNV,
    GetPathColorGenfvNV, GetPathTexGenivNV and GetPathTexGenfvNV:

        PATH_GEN_MODE_NV                                0x90B0
        PATH_GEN_COEFF_NV                               0x90B1

    Accepted by the <pname> parameter of GetPathColorGenivNV and
    GetPathColorGenfvNV:

        PATH_GEN_COLOR_FORMAT_NV                        0x90B2

    Accepted by the <pname> parameter of GetPathTexGenivNV and
    GetPathTexGenfvNV:

        PATH_GEN_COMPONENTS_NV                          0x90B3

    Accepted by the <programInterface> parameter of GetProgramInterfaceiv,
    GetProgramResourceIndex, GetProgramResourceName, GetProgramResourceiv,
    GetProgramResourcefvNV, and GetProgramResourceLocation:

        FRAGMENT_INPUT_NV                               0x936D

    Accepted in the <props> array of GetProgramResourceiv:

        PATH_GEN_MODE_NV                                see above
        PATH_GEN_COMPONENTS_NV                          see above

    Accepted in the <props> array of GetProgramResourcefvNV:

        PATH_GEN_COEFF_NV                               see above

Additions to Chapter 2 of the OpenGL 3.2 (unabridged) Specification
(OpenGL Operation)

    Add to the end of Section 2.12.1 (Matrices) with
    EXT_direct_state_access language applied...

    "The command

        void MatrixLoad3x2fNV(enum matrixMode, const float *m);

    is equivalent to:

        const float equiv_3x2matrix[16] = {
            m[0], m[2], 0, m[4],
            m[1], m[3], 0, m[5],
            0,    0,    1, 0,
            0,    0,    0, 1
        };
        MatrixLoadTransposefEXT(matrixMode, equiv_3x2matrix);

    The command

        void MatrixLoad3x3fNV(enum matrixMode, const float *m);

    is equivalent to:

        const float equiv_3x3matrix[16] = {
            m[0], m[3], 0, m[6],
            m[1], m[4], 0, m[7],
            0,    0,    1, 0,
            m[2], m[5], 0, m[8],
        };
        MatrixLoadTransposefEXT(matrixMode, equiv_3x3matrix);

    The command

        void MatrixLoadTranspose3x3fNV(enum matrixMode, const float *m);

    is equivalent to:

        const float equiv_3x3matrix[16] = {
            m[0], m[1], 0, m[2],
            m[3], m[4], 0, m[5],
            0,    0,    1, 0,
            m[6], m[7], 0, m[8],
        };
        MatrixLoadTransposefEXT(matrixMode, equiv_3x3matrix);

    The command

        void MatrixMult3x2fNV(enum matrixMode, const float *m);

    is equivalent to:

        const float equiv_3x2matrix[16] = {
            m[0], m[2], 0, m[4],
            m[1], m[3], 0, m[5],
            0,    0,    1, 0,
            0,    0,    0, 1
        };
        MatrixMultTransposefEXT(matrixMode, equiv_3x2matrix);

    The command

        void MatrixMult3x3fNV(enum matrixMode, const float *m);

    is equivalent to:

        const float equiv_3x3matrix[16] = {
            m[0], m[3], 0, m[6],
            m[1], m[4], 0, m[7],
            0,    0,    1, 0,
            m[2], m[5], 0, m[8],
        };
        MatrixMultTransposefEXT(matrixMode, equiv_3x3matrix);

    The command

        void MatrixMultTranspose3x3fNV(enum matrixMode, const float *m);

    is equivalent to:

        const float equiv_3x3matrix[16] = {
            m[0], m[1], 0, m[2],
            m[3], m[4], 0, m[5],
            0,    0,    1, 0,
            m[6], m[7], 0, m[8],
        };
        MatrixMultTransposefEXT(matrixMode, equiv_3x3matrix);"

    Modify the ARB_program_interface_query language as follows...

    Add to the "query properties of the interfaces of a program object"
    paragraph, so the "supported values of <programInterface>" includes:

      * FRAGMENT_INPUT_NV corresponds to the set of active input variables
        used by the fragment shader stage of <program> (if a fragment
        stage exists).  (This may be different from PROGRAM_INPUT except
        when the first shader stage is the fragment stage when they will
        be identical.)

    Change this sentence about when locations are assigned to include
    FRAGMENT_INPUT_NV so it reads:

    "When a program is linked successfully, active variables in the
    UNIFORM, PROGRAM_INPUT, FRAGMENT_INPUT_NV, PROGRAM_OUTPUT interface,
    or in any of the subroutine uniform interfaces, are assigned one or
    more signed integer /locations/."

    Amend Table X.1 "GetProgramResourceiv properties and supported
    interfaces" to add FRAGMENT_INPUT_NV to all the properties that
    allow PROGRAM_INPUT.  Specifically:

      * NAME_LENGTH
      * TYPE
      * ARRAY_SIZE
      * REFERENCED_BY_*_SHADER
      * LOCATION
      * IS_PER_PATCH (will always be false for FRAGMENT_INPUT_NV)

    Further amend Table X.1 "GetProgramResourceiv properties with two
    more properties for fragment input path generation state:

      Property                     Supported Interfaces
      ---------------------------  ----------------------------------------
      PATH_GEN_MODE_NV             FRAGMENT_INPUT_NV
      PATH_GEN_COMPONENTS_NV       FRAGMENT_INPUT_NV

    Amend the discussion of GetProgramResourceiv properties, adding:

    "For the property PATH_GEN_MODE_NV, a single integer identifying
    the path generation mode of an active variable is written to
    <params>.  The integer returned is one of NONE, OBJECT_LINEAR,
    PATH_OBJECT_BOUNDING_BOX_NV, or EYE_LINEAR based on how
    ProgramPathFragmentInputGenNV last specified the program resource's
    path generation mode.  The initial state is NONE.

    For the property PATH_GEN_COMPONENTS_NV, a single integer identifying
    the number of generated path components an active variable is written
    to <params>.  The integer returned is between 0 and 4 based on how
    ProgramPathFragmentInputGenNV last specified the program resource's
    path generation number of components.  The initial state is 0."

    Amend the list of tokens supported for the <programInterface> parameter of
    GetProgramResourceLocation to include FRAGMENT_INPUT_NV so the relevant
    sentence reads:

    "For GetProgramResourceLocation, <programInterface> must be one
    of UNIFORM, PROGRAM_INPUT, FRAGMENT_INPUT_NV, PROGRAM_OUTPUT,
    VERTEX_SUBROUTINE_UNIFORM, TESS_CONTROL_SUBROUTINE_UNIFORM,
    TESS_EVALUATION_SUBROUTINE_UNIFORM, GEOMETRY_SUBROUTINE_UNIFORM,
    FRAGMENT_SUBROUTINE_UNIFORM, or COMPUTE_SUBROUTINE_UNIFORM."

    After the discussion of GetProgramResourceiv, add:

    "The command

        void GetProgramResourcefvNV(uint program, enum programInterface,
                                    uint index, sizei propCount,
                                    const enum *props, sizei bufSize,
                                    sizei *length, float *params);

    operates in the same manner as GetProgramResourceiv expect the
    returned parameters values are floating-point and the only valid
    value of <programInterface> is FRAGMENT_INPUT_NV and the only valid value
    for the elements of the <props> array is PATH_GEN_COEFF_NV; otherwise
    INVALID_ENUM is generated.

    For the property PATH_GEN_COEFF_NV, sixteen floating-point values
    are written to <params> (limited to writing <bufSize> floating-point
    values)."

Additions to Chapter 3 of the OpenGL 3.2 (unabridged) Specification
(Rasterization)

    Append to the end of the "Shader Inputs" subsection of Section 3.12.2
    "Shader Execution":

    The command

        void ProgramPathFragmentInputGenNV(uint program,
                                           int location,
                                           enum genMode,
                                           int components,
                                           const float *coeffs);

    controls how a user-defined (non-built-in) fragment input of a
    GLSL program object is computed for fragment shading operations that
    occur as a result of CoverFillPathNV or CoverStrokePathNV.

    /program/ names a GLSL program object.  If /program/ has not been
    successfully linked, the error INVALID_OPERATION is generated.

    The given fragment input generation state is loaded into the fragment
    input variable location identified by /location/.  This location
    is a value returned either by GetProgramResourceLocation with a
    /programInterface/ of FRAGMENT_INPUT_NV and a given fragment shader
    input variable name or by GetProgramResourceiv with FRAGMENT_INPUT_NV
    for the /programInterface/ and LOCATION for the property for a given
    fragment input resource index.

    If the value of location is -1, the ProgramPathFragmentInputGenNV command
    will silently ignore the command, and the program's path fragment input
    generation state will not be changed.

    If any of the following conditions occur, an INVALID_OPERATION error
    is generated by the ProgramPathFragmentInputGenNV, and no state is changed:

        * if the size indicated in the /components/ of the
          ProgramPathFragmentInputGenNV command used does not match the
          size of the fragment input scalar or vector declared in the
          shader,

        * if the fragment input declared in the shader is not
          single-precision floating-point, or

        * if no fragment input variable with a location of /location/
          exists in the program object named by /program/ and location
          is not -1, or

        * if the fragment input declared in the shader is a built-in
          variables (i.e. prefixed by "gl_").

    When covering paths, fragment input variables are interpolated at
    each shaded fragment based on the corresponding fragment input
    generation state specified by ProgramPathFragmentInputGenNV for
    each respective fragment input.

    The /genMode/, /components/, and /coeffs/ parameters are used to
    generate the fragment input variable values identically as the
    PathTexGenNV command's corresponding parameters except it is a
    fragment input that is generated rather than a texture coordinate set
    (see the "TEXTURE COORDINATE SET GENERATION FOR PATH COVER COMMANDS"
    discussion in section 5.X.2.2 "Path Covering").  Because there is
    no associated texture coordinate set, the sc, tc, rc, and qc values
    when discussing PathTexGenNV are always zero when generating fragment
    input variables.

    When covering paths, if a fragment input variable has not had its
    path fragment input generation state successfully generated, it as
    if the values of this variable are always initialized to zero when
    the fragment shader is executing.

    Also when covering paths, GLSL fragment shaders support the following
    built-in fragment input variables:

        in vec4 gl_TexCoord[gl_MaxTextureCoords];
        in vec4 gl_Color
        in vec4 gl_FrontColor;
        in vec4 gl_BackColor;
        in vec4 gl_SecondaryColor;
        in vec4 gl_FrontSecondaryColor;
        in vec4 gl_BackSecondaryColor;
        in float gl_FogFragCoord;

    These respectively are initialized to the fragment input generated
    coordinates of PathTexGenNV, PathColorGenNV for GL_PRIMARY_COLOR_NV
    (front or back), PathColorGenNV for GL_SECONDARY_COLOR_NV (front or
    back), and glPathFogGenNV."

Additions to Chapter 4 of the OpenGL 3.2 (unabridged) Specification
(Per-Fragment Operations and the Frame Buffer)

    None

Additions to Chapter 5 of the OpenGL 3.2 (unabridged) Specification
(Special Functions)

 -- Insert section 5.X "Path Rendering" after 5.3 "Feedback"

    5.X Path Rendering

    5.X.1 Path Specification

    PATH COMMANDS

    Paths are specified as a sequence of path commands; each path command
    has an associated sequence of floating-point coordinates with the
    number of such coordinates depending on the specific path command.
    Coordinates are specified in a sequence independent from the path
    command sequence; coordinates from the coordinate sequence are matched
    up with (associated with) commands, in the order of the command,
    with coordinates extracted from the front of the coordinate sequence.

    Valid path commands are listed in table 5.pathCommands.  Each path
    command is listed with its associated token, description, character
    alias, count of associated coordinates.

    As an example of how path commands associated with path coordinates,
    if the command sequence was MOVE_TO_NV, LINE_TO_NV, CUBIC_CURVE_TO_NV,
    CLOSE_PATH_NV and the coordinates were 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, the MOVE_TO_NV command would be matched to coordinates 1 and 2,
    LINE_TO_NV would be matched to 3 and 4, CUBIC_CURVE_TO_NV would be
    matched to 5, 6, 7, 8, 9, 10, and CLOSE_PATH_NV would be matched to
    no coordinates.

    Path commands are processed in their sequence order to generate the
    path's outline.  The outline generation process maintains three 2D
    (x,y) state variables for each path processed: the start position
    (sp), the current position (cp), and the prior end point (pep);
    /sp/, /cp/ and /pep/ are initially (0,0) when a path starts being
    processed.

    Table 5.pathCommands: Path Commands

                                                       Character   Coordinate
    Token                       Description            alias       count
    ==========================  =====================  ==========  ==========
    MOVE_TO_NV                  Absolute move          'M'         2
                                current point
    RELATIVE_MOVE_TO_NV         Relative move          'm'         2
                                current point
    --------------------------  ---------------------  ----------  ----------
    CLOSE_PATH_NV               Close path             'Z' or 'z'  0
    RESTART_PATH_NV             Reset the path         -           0
    --------------------------  ---------------------  ----------  ----------
    LINE_TO_NV                  Absolute line          'L'         2
    RELATIVE_LINE_TO_NV         Relative line          'l'         2
    --------------------------  ---------------------  ----------  ----------
    HORIZONTAL_LINE_TO_NV       Absolute horizontal    'H'         1
                                line
    RELATIVE_HORIZONTAL-        Relative horizontal    'h'         1
      _LINE_TO_NV               line
    VERTICAL_LINE_TO_NV         Absolute vertical      'V'         1
                                line
    RELATIVE_VERTICAL-          Relative vertical      'v'         1
      _LINE_TO_NV               line
    --------------------------  ---------------------  ----------  ----------
    QUADRATIC_CURVE_TO_NV       Absolute quadratic     'Q'         4
                                Bezier segment
    RELATIVE-                   Relative quadratic     'q'         4
      _QUADRATIC_CURVE_TO_NV    Bezier segment
    --------------------------  ---------------------  ----------  ----------
    CUBIC_CURVE_TO_NV           Absolute cubic         'C'         6
                                Bezier segment
    RELATIVE_CUBIC_CURVE_TO_NV  Relative cubic         'c'         6
                                Bezier segment
    --------------------------  ---------------------  ----------  ----------
    SMOOTH-                     Absolute smooth        'T'         2
      _QUADRATIC_CURVE_TO_NV    quadratic Bezier
                                segment
    RELATIVE_SMOOTH-            Relative smooth        't'         2
      _QUADRATIC_CURVE_TO_NV    quadratic Bezier
                                segment
    --------------------------  ---------------------  ----------  ----------
    SMOOTH-                     Absolute smooth        'S'         4
      _CUBIC_CURVE_TO_NV        cubic Bezier segment
    RELATIVE_SMOOTH-            Relative smooth        's'         4
      _CUBIC_CURVE_TO_NV        cubic Bezier segment
    --------------------------  ---------------------  ----------  ----------
    SMALL_CCW_ARC_TO_NV         Absolute small-sweep   -           5
                                counterclockwise
                                partial elliptical
                                arc segment
    RELATIVE-                   Relative small-sweep   -           5
      _SMALL_CCW_ARC_TO_NV      counterclockwise
                                partial elliptical
                                arc segment
    SMALL_CW_ARC_TO_NV          Absolute small-sweep   -           5
                                clockwise partial
                                elliptical arc
                                segment
    RELATIVE-                   Relative small-sweep   -           5
      _SMALL_CW_ARC_TO_NV       clockwise partial
                                elliptical arc
                                segment
    LARGE_CCW_ARC_TO_NV         Absolute large-sweep   -           5
                                counterclockwise
                                partial elliptical
                                arc segment
    RELATIVE-                   Relative large-sweep   -           5
      _LARGE_CCW_ARC_TO_NV      counterclockwise
                                partial elliptical
                                arc segment
    LARGE_CW_ARC_TO_NV          Absolute large-sweep   -           5
                                clockwise partial
                                elliptical arc
                                segment
    RELATIVE-                   Relative large-sweep   -           5
      _LARGE_CW_ARC_TO_NV       clockwise partial
                                elliptical arc
                                segment
    --------------------------  ---------------------  ----------  ----------
    CONIC_CURVE_TO_NV           Absolute conic         'W'         5
                                (rational Bezier)
                                segment
    RELATIVE-                   Relative conic         'w'         5
      _CONIC_CURVE_TO_NV        (rational Bezier)
                                segment
    --------------------------  ---------------------  ----------  ----------
    ROUNDED_RECT_NV             Absolute rounded       -           5
                                rectangle with
                                uniform circular
                                corners (1 radius)
    RELATIVE_ROUNDED_RECT_NV    Relative rounded       -           5
                                rectangle with
                                uniform circular
                                corners (1 radius)
    ROUNDED_RECT2_NV            Absolute rounded       -           6
                                rectangle with
                                uniform elliptical
                                corners (2 x&y radii)
    RELATIVE_ROUNDED_RECT2_NV   Relative rounded       -           6
                                rectangle with
                                uniform elliptical
                                corners (2 x&y radii)
    ROUNDED_RECT4_NV            Absolute rounded       -           8
                                rectangle with
                                varying circular
                                corners (4 radii)
    RELATIVE_ROUNDED_RECT4_NV   Relative rounded       -           8
                                rectangle with
                                varying circular
                                corners (4 radii)
    ROUNDED_RECT8_NV            Absolute rounded       -           12
                                rectangle with
                                varying elliptical
                                corners (8 radii)
    RELATIVE_ROUNDED_RECT8_NV   Relative rounded       -           12
                                rectangle with
                                varying elliptical
                                corners (8 radii)
    --------------------------  ---------------------  ----------  ----------
    DUP_FIRST_-                 Absolute cubic Bezier  -           4
      CUBIC_CURVE_TO_NV         segment, duplicating
                                first control point
    DUP_LAST_CUBIC_CURVE_TO_NV  Absolute cubic Bezier  -           4
                                segment, duplicating
                                last control point
    RECT_NV                     Closed absolute        -           4
                                rectangle
    RELATIVE_RECT_NV            Closed relative        -           4
                                rectangle
    --------------------------  ---------------------  ----------  ----------
    CIRCULAR_CCW_ARC_TO_NV      Absolute               -           5
                                counterclockwise
                                circular arc segment
    CIRCULAR_CW_ARC_TO_NV       Absolute clockwise     -           5
                                circular arc segment
    CIRCULAR_TANGENT_ARC_TO_NV  Absolute circular      -           5
                                tangential
                                arc segment
    --------------------------  ---------------------  ----------  ----------
    ARC_TO_NV                   Absolute general       'A'         7
                                elliptical arc
    RELATIVE_ARC_TO_NV          Relative general       'a'         7
                                elliptical arc
    --------------------------  ---------------------  ----------  ----------

    Table 5.pathEquations provides for each path command, as relevant,
    the command's path segment parametric equation, equations for the
    updated current point (ncp) and equations for the updated prior
    end point (npep).  After each command in a path is processed in the
    sequence, the new current point, prior end point, and start point
    (if changed) update the current point, prior end point, and start
    point for the next path command to be processed in the sequence.  So:

       cp = ncp
       pep = npep

    Each path segment parametric equation is parameterized by a variable
    /t/ ranging from 0.0 to 1.0.  So the outline is traced by evaluating
    each path command's path segment parametric equation continuously
    as /t/ varies from 0.0 to 1.0.

    With the exception of the MOVE_TO_NV, RELATIVE_MOVE_TO_NV,
    RESTART_PATH_NV, RECT_NV, RELATIVE_RECT_NV, ROUNDED_RECT_NV,
    RELATIVE_ROUNDED_RECT_NV, ROUNDED_RECT2_NV, RELATIVE_ROUNDED_RECT2_NV,
    ROUNDED_RECT4_NV, RELATIVE_ROUNDED_RECT4_NV, ROUNDED_RECT8_NV,
    RELATIVE_ROUNDED_RECT8_NV, CIRCULAR_CCW_ARC_TO_NV, and
    CIRCULAR_CW_ARC_TO_NV commands, the commands are specified such that
    C0 continuity of the outline is guaranteed at path command segment
    end-points.

    The MOVE_TO_NV, RELATIVE_MOVE_TO_NV, RESTART_PATH_NV, RECT_NV,
    RELATIVE_RECT_NV, ROUNDED_RECT_NV, RELATIVE_ROUNDED_RECT_NV,
    ROUNDED_RECT2_NV, RELATIVE_ROUNDED_RECT2_NV,
    ROUNDED_RECT4_NV, RELATIVE_ROUNDED_RECT4_NV, ROUNDED_RECT8_NV,
    RELATIVE_ROUNDED_RECT8_NV, CIRCULAR_CCW_ARC_TO_NV, and
    CIRCULAR_CW_ARC_TO_NV commands update the start position (sp) to
    the value of these command's new current point (ncp).

    The MOVE_TO_NV, RELATIVE_MOVE_TO_NV, RECT_NV,
    RELATIVE_RECT_NV, ROUNDED_RECT_NV, RELATIVE_ROUNDED_RECT_NV,
    ROUNDED_RECT2_NV, RELATIVE_ROUNDED_RECT2_NV,
    ROUNDED_RECT4_NV, RELATIVE_ROUNDED_RECT4_NV, ROUNDED_RECT8_NV,
    RELATIVE_ROUNDED_RECT8_NV, commands unconditionally change the start
    position (sp) to value of these command's new current point (ncp) so:

        sp = ncp

    The CIRCULAR_CCW_ARC_TO_NV and CIRCULAR_CW_ARC_TO_NV commands
    conditionally change sp to the command's ncp but only the sp has not
    been specified by any prior command other than CLOSE_PATH_NV in the
    path's command sequence since the beginning of the path's command
    sequence or last RESTART_PATH_NV.  When these circular arc commands
    change the sp to the command's ncp, it implies the initial implicit
    line these commands generate from sp to ncp will be zero length.
    (This behavior is to match the semantics of PostScript.)

    Moving of the start position creates a discontinuity in the outline
    so starts a new subpath within the path.

    Table 5.pathEquations: Path Equations

                                Path segment                            new current         new prior end
    Token                       parametric equation                     point equation      point equation
    ==========================  ======================================  ==================  =======================
    MOVE_TO_NV                  -                                       ncp.x = c[0]        npep.x = c[0]
                                                                        ncp.y = c[1]        npep.y = c[1]
    RELATIVE_MOVE_TO_NV         -                                       ncp.x = cp.x+c[0]   npep.x = cp.x+c[0]
                                                                        ncp.y = cp.y+c[1]   npep.y = cp.y+c[1]
    --------------------------  --------------------------------------  ------------------  -----------------------
    CLOSE_PATH_NV               x = (1-t)*cp.x + t*sp.x                 ncp.x = sp.x        npep.x = sp.x
                                y = (1-t)*cp.y + t*sp.y                 ncp.y = sp.y        npep.y = sp.y
    RESTART_PATH_NV             -                                       ncp.x = 0           npep.x = 0
                                                                        ncp.y = 0           npep.y = 0
    --------------------------  --------------------------------------  ------------------  -----------------------
    LINE_TO_NV                  x = (1-t)*cp.x + t*c[0]                 ncp.x = c[0]        npep.x = c[0]
                                y = (1-t)*cp.y + t*c[1]                 ncp.y = c[1]        npep.y = c[1]
    RELATIVE_LINE_TO_NV         x = (1-t)*cp.x + t*(c[0]+cp.x)          ncp.x = cp.x+c[0]   npep.x = cp.x+c[0]
                                y = (1-t)*cp.y + t*(c[1]+cp.y)          ncp.y = cp.y+c[1]   npep.y = cp.y+c[1]
    --------------------------  --------------------------------------  ------------------  -----------------------
    HORIZONTAL_LINE_TO_NV       x = (1-t)*cp.x + t*sp.x                 ncp.x = c[0]        npep.x = c[0]
                                y = cp.y                                ncp.y = cp.y        npep.y = cp.y
    RELATIVE_HORIZONTAL-        x = (1-t)*cp.x + t*(c[0]+cp.x)          ncp.x = cp.x+c[0]   npep.x = cp.x+c[0]
      _LINE_TO_NV               y = cp.y                                ncp.y = cp.y        npep.y = cp.y
    VERTICAL_LINE_TO_NV         x = cp.x                                ncp.x = cp.x        npep.x = cp.x
                                y = (1-t)*cp.y + t*sp.y                 ncp.y = c[0]        npep.y = c[0]
    RELATIVE_VERTICAL-          x = cp.x                                ncp.x = cp.x        npep.x = cp.x
      _LINE_TO_NV               y = (1-t)*cp.y + t*(c[0]+cp.y)          ncp.y = cp.y+c[0]   npep.y = cp.y+c[0]
    --------------------------  --------------------------------------  ------------------  -----------------------
    QUADRATIC_CURVE_TO_NV       x = (1-t)^2*cp.x +                      ncp.x = c[2]        npep.x = c[0]
                                    2*(1-t)*t*c[0] +                    ncp.y = c[3]        npep.y = c[1]
                                    t^2*c[2]
                                y = (1-t)^2*cp.y +
                                    2*(1-t)*t*c[1] +
                                    t^2*c[3]
    RELATIVE-                   x = (1-t)^2*cp.x +                      ncp.x = cp.x+c[2]   npep.x = cp.x+c[0]
      _QUADRATIC_CURVE_TO_NV        2*(1-t)*t*(c[0]+cp.x) +             ncp.y = cp.x+c[3]   npep.y = cp.y+c[1]
                                    t^2*(c[2]+cp.x)
                                y = (1-t)^2*cp.y +
                                    2*(1-t)*t*(c[1]+cp.y) +
                                    t^2*(c[3]+cp.y)
    --------------------------  --------------------------------------  ------------------  -----------------------
    CUBIC_CURVE_TO_NV           x = (1-t)^3*cp.x +                      ncp.x = c[4]        npep.x = c[2]
                                    3*(1-t)^2*t*c[0] +                  ncp.y = c[5]        npep.y = c[3]
                                    3*(1-t)*t^2*c[2] +
                                    t^3*c[4]
                                y = (1-t)^3*cp.y +
                                    3*(1-t)^2*t*c[1] +
                                    3*(1-t)*t^2*c[3] +
                                    t^3*c[5]
    RELATIVE_CUBIC_CURVE_TO_NV  x = (1-t)^3*cp.x +                      ncp.x = cp.x+c[4]   npep.x = cp.x+c[2]
                                    3*(1-t)^2*t*(c[0]+cp.x) +           ncp.y = cp.y+c[5]   npep.y = cp.y+c[3]
                                    3*(1-t)*t^2*(c[2]+cp.x) +
                                    t^3*(c[4]+cp.x)
                                y = (1-t)^3*cp.y +
                                    3*(1-t)^2*t*(c[1]+cp.y) +
                                    3*(1-t)*t^2*(c[3]+cp.y) +
                                    t^3*(c[5]+cp.y)
    --------------------------  --------------------------------------  ------------------  -----------------------
    SMOOTH-                     x = (1-t)^2*cp.x +                      ncp.x = c[0]        npep.x = 2*cp.x-pep.x
      _QUADRATIC_CURVE_TO_NV        2*(1-t)*t*(2*cp.x-pep.x) +          ncp.y = c[1]        npep.y = 2*cp.y-pep.y
                                    t^2*c[0]
                                y = (1-t)^2*cp.y +
                                    2*(1-t)*t*(2*cp.y-pep.y) +
                                    t^2*c[1]
    RELATIVE_SMOOTH-            x = (1-t)^2*cp.x +                      ncp.x = cp.x+c[0]   npep.x = 2*cp.x-pep.x
      QUADRATIC_CURVE_TO_NV         2*(1-t)*t*(2*cp.x-pep.x) +          ncp.y = cp.y+c[1]   npep.y = 2*cp.y-pep.y
                                    t^2*(c[0]+cp.x)
                                y = (1-t)^2*cp.y +
                                    2*(1-t)*t*(2*cp.y-pep.y) +
                                    t^2*(c[1]+cp.y)

    SMOOTH-                     x = (1-t)^3*cp.x +                      ncp.x = c[2]        npep.x = c[0]
      _CUBIC_CURVE_TO_NV            3*(1-t)^2*t*(2*cp.x-pep.x) +        ncp.y = c[3]        npep.y = c[1]
                                    3*(1-t)*t^2*c[0] +
                                    t^3*c[2]
                                y = (1-t)^3*cp.y +
                                    3*(1-t)^2*t*(2*cp.y-pep.y) +
                                    3*(1-t)*t^2*c[1] +
                                    t^3*c[3]
    RELATIVE_SMOOTH-            x = (1-t)^3*cp.x +                      ncp.x = cp.x+c[2]   npep.x = cp.x+c[0]
      _CUBIC_CURVE_TO_NV            3*(1-t)^2*t*(2*cp.x-pep.x) +        ncp.y = cp.y+c[3]   npep.y = cp.y+c[1]
                                    3*(1-t)*t^2*(c[0]+cp.x) +
                                    t^3*(c[2]+cp.x)
                                y = (1-t)^3*cp.y +
                                    3*(1-t)^2*t*(2*cp.y-pep.y) +
                                    3*(1-t)*t^2*(c[1]+cp.y) +
                                    t^3*(c[3]+cp.y)
    --------------------------  --------------------------------------  ------------------  -----------------------
    SMALL_CCW_ARC_TO_NV         x = arc_x(c,rv,rh,phi,                  ncp.x = c[3]        npep.x = c[3]
                                          theta1,dtheta,t)              ncp.y = c[4]        npep.y = c[4]
                                y = arc_y(c,rv,rh,phi,
                                          theta1,dtheta,t)
    RELATIVE-                   x = arc_x(c,rv,rh,phi,                  ncp.x = c[3]        npep.x = cp.x+c[3]
      _SMALL_CCW_ARC_TO_NV                theta1,dtheta,t)              ncp.y = c[4]        npep.y = cp.y+c[4]
                                y = arc_y(c,rv,rh,phi,
                                          theta1,dtheta,t)
    SMALL_CW_ARC_TO_NV          x = arc_x(c,rv,rh,phi,                  ncp.x = c[3]        npep.x = c[3]
                                          theta1,dtheta,t)              ncp.y = c[4]        npep.y = c[4]
                                y = arc_y(c,rv,rh,phi,
                                          theta1,dtheta,t)
    RELATIVE-                   x = arc_x(c,rv,rh,phi,                  ncp.x = c[3]        npep.x = cp.x+c[3]
      _SMALL_CW_ARC_TO_NV                 theta1,dtheta,t)              ncp.y = c[4]        npep.y = cp.y+c[4]
                                y = arc_y(c,rv,rh,phi,
                                          theta1,dtheta,t)
    LARGE_CCW_ARC_TO_NV         x = arc_x(c,rv,rh,phi,                  ncp.x = c[3]        npep.x = c[3]
                                          theta1,dtheta,t)              ncp.y = c[4]        npep.y = c[4]
                                y = arc_y(c,rv,rh,phi,
                                          theta1,dtheta,t)
    RELATIVE-                   x = arc_x(c,rv,rh,phi,                  ncp.x = c[3]        npep.x = cp.x+c[3]
      _LARGE_CCW_ARC_TO_NV                theta1,dtheta,t)              ncp.y = c[4]        npep.y = cp.y+c[4]
                                y = arc_y(c,rv,rh,phi,
                                          theta1,dtheta,t)
    LARGE_CW_ARC_TO_NV          x = arc_x(c,rv,rh,phi,                  ncp.x = c[3]        npep.x = c[3]
                                          theta1,dtheta,t)              ncp.y = c[4]        npep.y = c[4]
                                y = arc_y(c,rv,rh,phi,
                                          theta1,dtheta,t)
    RELATIVE-                   x = arc_x(c,rv,rh,phi,                  ncp.x = c[3]        npep.x = cp.x+c[3]
      _SMALL_CW_ARC_TO_NV                 theta1,dtheta,t)              ncp.y = c[4]        npep.y = cp.y+c[4]
                                y = arc_y(c,rv,rh,phi,
                                          theta1,dtheta,t)
    --------------------------  --------------------------------------  ------------------  -----------------------
    CONIC_CURVE_TO_NV           WHEN c[4] > 0:                          ncp.x = c[2]        npep.x = c[0]
                                x = ( (1-t)^2*cp.x +                    ncp.y = c[3]        npep.y = c[1]
                                      2*(1-t)*t*c[0]*c[4] +
                                      t^2*c[2] ) /
                                    ( (1-t)^2 +
                                      2*(1-t)*t*c[4] +
                                      t^2*c[2] )
                                y = ( 1-t)^2*cp.y +
                                      2*(1-t)*t*c[1]*w +
                                      t^2*c[3] ) /
                                    ( (1-t)^2 +
                                      2*(1-t)*t*c[4] +
                                      t^2*c[2] ),
                                OTHERWISE:
                                x = (1-t)*cp.x + t*c[2]
                                y = (1-t)*cp.y + t*c[3]
    RELATIVE-                   WHEN c[4] > 0:                          ncp.x = cp.x+c[2]   npep.x = cp.x+c[0]
      _CONIC_CURVE_TO_NV        x = ( (1-t)^2*cp.x +                    ncp.y = cp.y+c[3]   npep.y = cp.y+c[1]
                                      2*(1-t)*t*(c[0]+cp.x)*c[4] +
                                      t^2*(c[2]+cp.x) ) /
                                    ( (1-t)^2 +
                                      2*(1-t)*t*c[4] +
                                      t^2*c[2] )
                                y = ( 1-t)^2*cp.y +
                                      2*(1-t)*t*c[1]*w +
                                      t^2*c[3] ) /
                                    ( (1-t)^2 +
                                      2*(1-t)*t*c[4] +
                                      t^2*c[2] ),
                                OTHERWISE:
                                x = (1-t)*cp.x + t*(c[2]+cp.x)
                                y = (1-t)*cp.y + t*(c[3]+cp.y)
    --------------------------  --------------------------------------  ------------------  -----------------------
    ROUNDED_RECT_NV             x = rrect(c[0], c[1], c[2], c[3],       ncp.x = C_ll.x      npep.x = C_ll.x
                                          c[4], c[4], c[4], c[4],       ncp.y = C_ll.y      npep.y = C_ll.y
                                          c[4], c[4], c[4], c[4], t).x
                                y = rrect(c[0], c[1], c[2], c[3],
                                          c[4], c[4], c[4], c[4],
                                          c[4], c[4], c[4], c[4], t).y
    RELATIVE_ROUNDED_RECT_NV    x = rrect(c[0]+cp.x, c[1]+cp.y,         ncp.x = C_ll.x      npep.x = C_ll.x
                                          c[2], c[3],                   ncp.y = C_ll.y      npep.y = C_ll.y
                                          c[4], c[4], c[4], c[4],
                                          c[4], c[4], c[4], c[4], t).x
                                y = rrect(c[0]+cp.x, c[1]+cp.y,
                                          c[2], c[3],
                                          c[4], c[4], c[4], c[4],
                                          c[4], c[4], c[4], c[4], t).y
    ROUNDED_RECT2_NV            x = rrect(c[0], c[1], c[2], c[3],       ncp.x = C_ll.x      npep.x = C_ll.x
                                          c[4], c[5], c[4], c[5],       ncp.y = C_ll.y      npep.y = C_ll.y
                                          c[4], c[5], c[4], c[5], t).x
                                y = rrect(c[0], c[1], c[2], c[3],
                                          c[4], c[5], c[4], c[5],
                                          c[4], c[5], c[4], c[5], t).y
    RELATIVE_ROUNDED_RECT2_NV   x = rrect(c[0]+cp.x, c[1]+cp.y,         ncp.x = C_ll.x      npep.x = C_ll.x
                                          c[2], c[3],                   ncp.y = C_ll.y      npep.y = C_ll.y
                                          c[4], c[5], c[4], c[5],
                                          c[4], c[5], c[4], c[5], t).x
                                y = rrect(c[0]+cp.x, c[1]+cp.y,
                                          c[2], c[3],
                                          c[4], c[5], c[4], c[5],
                                          c[4], c[5], c[4], c[5], t).y
    ROUNDED_RECT4_NV            x = rrect(c[0], c[1], c[2], c[3],       ncp.x = C_ll.x      npep.x = C_ll.x
                                          c[4], c[4], c[5], c[5],       ncp.y = C_ll.y      npep.y = C_ll.y
                                          c[6], c[6], c[7], c[7], t).x
                                y = rrect(c[0], c[1], c[2], c[3],
                                          c[4], c[4], c[5], c[5],
                                          c[6], c[6], c[7], c[7], t).y
    RELATIVE_ROUNDED_RECT4_NV   x = rrect(c[0]+cp.x, c[1]+cp.y,         ncp.x = C_ll.x      npep.x = C_ll.x
                                          c[2], c[3],                   ncp.y = C_ll.y      npep.y = C_ll.y
                                          c[4], c[4], c[5], c[5],
                                          c[6], c[6], c[7], c[7], t).x
                                y = rrect(c[0]+cp.x, c[1]+cp.y,
                                          c[2], c[3],
                                          c[4], c[4], c[5], c[5],
                                          c[6], c[6], c[7], c[7], t).y
    ROUNDED_RECT8_NV            x = rrect(c[0], c[1], c[2], c[3],       ncp.x = C_ll.x      npep.x = C_ll.x
                                          c[4], c[5], c[6], c[7],       ncp.y = C_ll.y      npep.y = C_ll.y
                                          c[8], c[9],
                                          c[10], c[11], t).x
                                y = rrect(c[0], c[1], c[2], c[3],
                                          c[4], c[5], c[6], c[7],
                                          c[8], c[9],
                                          c[10], c[11], t).y
    RELATIVE_ROUNDED_RECT8_NV   x = rrect(c[0]+cp.x, c[1]+cp.y,         ncp.x = C_ll.x      npep.x = C_ll.x
                                          c[2], c[3],                   ncp.y = C_ll.y      npep.y = C_ll.y
                                          c[4], c[5], c[6], c[7],
                                          c[8], c[9],
                                          c[10], c[11], t).x
                                y = rrect(c[0]+cp.x, c[1]+cp.y,
                                          c[2], c[3],
                                          c[4], c[5], c[6], c[7],
                                          c[8], c[9],
                                          c[10], c[11], t).y
    --------------------------  --------------------------------------  ------------------  -----------------------
    DUP_FIRST-                  x = (1-t)^3*cp.x +                      ncp.x = c[2]        npep.x = c[0]
      CUBIC_CURVE_TO_NV             3*(1-t)^2*t*cp.x +                  ncp.y = c[3]        npep.y = c[1]
                                    3*(1-t)*t^2*c[0] +
                                    t^3*c[2]
                                y = (1-t)^3*cp.y +
                                    3*(1-t)^2*t*cp.y +
                                    3*(1-t)*t^2*c[1] +
                                    t^3*c[3]
    DUP_LAST_CUBIC_CURVE_TO_NV  x = (1-t)^3*cp.x +                      ncp.x = c[2]        npep.x = c[2]
                                    3*(1-t)^2*t*c[0] +                  ncp.y = c[3]        npep.y = c[3]
                                    3*(1-t)*t^2*c[2] +
                                    t^3*c[2]
                                y = (1-t)^3*cp.y +
                                    3*(1-t)^2*t*c[1] +
                                    3*(1-t)*t^2*c[3] +
                                    t^3*c[3]
    --------------------------  --------------------------------------  ------------------  -----------------------
    RECT_NV                         / (1-4*t)*c[0] +                    ncp.x = c[0]        npep.x = c[0]
                                    | 4*t*(c[0]+c[2]),         t<=0.25  ncp.y = c[1]        npep.y = c[1]
                                x = < c[0]+c[2],           0.25<t<=0.5
                                    | (1-4*t-2)*(c[0]+c[2]) +
                                    | (4*t-2)*c[0],       0.5 <t<=0.75
                                    \ c[0],                     0.75<t
                                    / c[1],                    t<=0.25
                                    | (1-4*t-1)*c[1] +
                                y = < (4*t-1)*(c[1]+c[3]), 0.25<t<=0.5
                                    | c[1]+c[3],          0.5 <t<=0.75
                                    | (1-4*t-3)*(c[1]+c[3]) +
                                    \ (4*t-3)*c[1],             0.75<t
    RELATIVE_RECT_NV                / (1-4*t)*(c[0]+cp.x) +             ncp.x = cp.x+c[0]   npep.x = cp.x+c[0]
                                    | 4*t*(c[0]+c[2]+cp.x),    t<=0.25  ncp.y = cp.y+c[1]   npep.y = cp.y+c[1]
                                x = < c[0]+c[2]+cp.x,      0.25<t<=0.5
                                    | (1-4*t-2)*(c[0]+c[2]+cp.x) +
                                    | (4*t-2)*(c[0]+cp.x), 0.5<t<=0.75
                                    \ c[0]+cp.x,                0.75<t
                                    / c[1]+cp.y,               t<=0.25
                                    | (1-4*t-1)*(c[1]+cp.y) +
                                y = < (4*t-1)*(c[1]+c[3]+cp.y),
                                    |                      0.25<t<=0.5
                                    | c[1]+c[3]+cp.y,      0.5<t<=0.75
                                    | (1-4*t-3)*(c[1]+c[3]+cp.y) +
                                    \ (4*t-3)*(c[1]+cp.y),      0.75<t
    --------------------------  --------------------------------------  ------------------  -----------------------
    CIRCULAR_CCW_ARC_TO_NV           / (1-2*t)*cp.x + 2*t*A.x,  t<=0.5  ncp.x = B.x         npep.x = B.x
                                x = {                                   ncp.y = B.y         npep.x = B.y
                                     \ arc_x(c,rv,rh,phi,theta1,
                                      \      dtheta,t*2-1)      t>=0.5
                                     / (1-2*t)*cp.y + 2*t*A.y,  t<=0.5
                                y = {
                                     \ arc_y(c,rv,rh,phi,theta1,
                                      \      dtheta,t*2-1),     t>=0.5
    CIRCULAR_CW_ARC_TO_NV            / (1-2*t)*cp.x + 2*t*A.x,  t<=0.5  ncp.x = B.x         npep.x = B.x
                                x = {                                   ncp.y = B.y         npep.x = B.y
                                     \ arc_x(c,rv,rh,phi,theta1,
                                      \      dtheta,t*2-1)      t>=0.5
                                     / (1-2*t)*cp.y + 2*t*A.y,  t<=0.5
                                y = {
                                     \ arc_y(c,rv,rh,phi,theta1,
                                      \      dtheta,t*2-1),     t>=0.5
    CIRCULAR_TANGENT_ARC_TO_NV       / (1-2*t)*cp.x + 2*t*C.x,  t<=0.5  ncp.x = D.x         npep.x = D.x
                                x = {                                   ncp.y = D.y         npep.x = D.y
                                     \ arc_x(c,rv,rh,phi,theta1,
                                      \      dtheta,t*2-1),     t>=0.5
                                     / (1-2*t)*cp.y + 2*t*C.y,  t<=0.5
                                y = {
                                     \ arc_y(c,rv,rh,phi,theta1,
                                      \      dtheta,t*2-1),     t>=0.5
    --------------------------  --------------------------------------  ------------------  -----------------------
    ARC_TO_NV                   x = arc_x(c,rv,rh,phi,                  ncp.x = c[5]        npep.x = c[5]
                                          theta1,dtheta,t)              ncp.y = c[6]        npep.y = c[6]
                                y = arc_y(c,rv,rh,phi,
                                          theta1,dtheta,t)
    RELATIVE_ARC_TO_NV          x = arc_x(c,rv,rh,phi,                  ncp.x = cp.x+c[5]   npep.x = cp.x+c[5]
                                          theta1,dtheta,t)              ncp.y = cp.y+c[6]   npep.y = cp.y+c[6]
                                y = arc_y(c,rv,rh,phi,
                                          theta1,dtheta,t)
    --------------------------  --------------------------------------  ------------------  -----------------------

    In the equations in Table 5.pathEquations, c[i] is the /i/th (base
    zero) coordinate of the coordinate sequence for the command; /cp/
    is the 2D (x,y) current position from the prior command (for the
    first command of a path object, /cp/ is (0,0)); /sp/ is the 2D (x,y)
    start position for the current contour (for the first command of a
    path object, /sp/ is (0,0)); /pep/ is the 2D (x,y) prior end position
    from the prior end position (for the first command of a path object,
    /pep/ is (0,0)); and /ncp/ is the 2D (x,y) "new" current position
    that will become the current position for the subsequent command;
    /npep/ is the 2D (x,y) "new" prior end position for the subsequent
    command.  The values /c/, /theta1/, /dtheta/ are explained in the
    discussion of partial elliptical arc commands below.  The values
    of /rv/, /rh/, /phi/ come from Table 5.arcParameterSpecialization.
    The values of /A/, /B/, /C/, and /D/ are discussed in the context
    of Table 5.arcParameterSpecialization.  /C_ll/ is the lower-left
    end-point defined for Equation 5.roundedRectangleContour.

    If a value specified for a coordinate (however the coordinate is
    specified) or a value computed from these coordinates (as specified
    in the discussion that follows) exceeds the implementation's maximum
    representable value for a single-precision floating-point number,
    the rendering behavior (discussed in section 5.X.2) of the specified
    path and the value of said coordinate if queried (section 6.X.2)
    is undefined.  This is relevant because coordinates can be specified
    explicitly but also relatively (by RELATIVE_* path commands) or
    encoded in a string of otherwise arbitrary precision and range or
    computed by weighting.

    ROUNDED RECTANGLE COMMAND DETAILS

    In all the rounded-rectangle path commands, the parametric segment
    path equations in Table 5.pathEquations are expressed in terms of
    the function /rrect/ that returns an (x,y) position on the rounded
    rectangle when evaluated over the parametric range t=0..1.  /rrect/
    is a spline of alternating rational quadratic Bezier segments and
    linear segments that forms a closed contour.

    In addition to its parametric variable /t/, /rrect/ has 12
    parameters to which geometric properties can be ascribed when these
    values are positive.  (No restriction precludes these parameters
    to be non-positive; negative and zero values are allowed and
    supported.) (x,y) is the 2D location of the lower-left corner of the
    rectangle tightly bounding the rounded rectangle contour; /w/ and /h/
    are the respective width and height of the same bounding rectangle.
    /r_llx/, /r_lrx/, /r_urx/, and /r_ulx/ are the elliptical x-axis
    radii corresponding to the lower-left, lower-right, upper-right,
    and upper-left corners of the rounded rectangle; likewise /r_lly/,
    /r_lry/, /r_ury/, and /r_uly/ are the elliptical y-axis for the
    same corners.

    Equation 5.roundedRectangleContour

        rrect(x,y,             / (1-t_0)*C_ll + t_0*A_lr,                                   t=0/8..1/8
              w,h,             |
              r_llx,r_lly,     | (1-t_1)^2*A_lr + (1-t_1)*t_1*B_lr*sqrt(2) + t_1^2*C_lr)/
              r_lrx,r_lry,  =  < ((1-t_1)^2 + (1-t_1)*t_1*sqrt(2) + t_1^2),                 t=1/8..2/8
              r_urx,r_ury,     |
              r_ulx,r_uly,     | (1-t_2)*C_lr + t_2*A_ur,                                   t=2/8..3/8
              t)               |
                               | (1-t_3)^2*A_ur + (1-t_3)*t_3*B_ur*sqrt(2) + t_3^2*C_ur)/
                               | ((1-t_3)^2 + (1-t_3)*t_3*sqrt(2) + t_3^2),                 t=3/8..4/8
                               |
                               | (1-t_4)*C_ur + t_4*A_ul,                                   t=4/8..5/8
                               |
                               | (1-t_5)^2*A_ul + (1-t_5)*t_5*B_ul*sqrt(2) + t_5^2*C_ul)/
                               | ((1-t_5)^2 + (1-t_5)*t_5*sqrt(2) + t_5^2),                 t=5/8..6/8
                               |
                               | (1-t_6)*C_ul + t_6*A_ll,                                   t=6/8..7/8
                               |
                               | (1-t_7)^2*A_ll + (1-t_7)*t_7*B_ll*sqrt(2) + t_7^2*C_ll)/
                               \ ((1-t_7)^2 + (1-t_7)*t_7*sqrt(2) + t_7^2),                 t=7/8..1

    where

        t_i = 8*t - i

        A_ll = (x,y-r_lly), h<0
               (x,y+r_lly), otherwise
        B_ll = (x,y)
        C_ll = (x-r_llx,y), w<0
               (x+r_llx,y), otherwise

        A_lr = (x+w+r_lrx,y), w<0
               (x+w-r_lrx,y), otherwise
        B_lr = (x+w,y)
        C_lr = (x+w,y-r_lry), h<0
               (x+w,y+r_lry), otherwise

        A_ur = (x+w,y+h-r_ury*sign(h)), h<0
               (x+w,y+h-r_ury*sign(h)), otherwise
        B_ur = (x+w,y+h)
        C_ur = (x+w+r_urx,y+h), w<0
               (x+w-r_urx,y+h), otherwise

        A_ul = (x-r_ulx,y+h), w<0
               (x+r_ulx,y+h), otherwise
        B_ul = (x,y+h)
        C_ul = (x,y+h+r_uly), h<0
               (x,y+h-r_uly), otherwise

    Consider /t_i/ to be a subparmetric range within /t/ where /t_i/
    ranges over 0..1 for each spline segment of /rrect/.  The 2D control
    points /A_ll/ through /C_ul/ are shown in Figure 5.roundedRectangle,
    where /ll/, /lr/, /ur/, and /ul/ correspond to lower-left,
    lower-right, upper-right, and upper-left respectively.

    Figure 5.roundedRectangle:  Geometric interpretation of Equation
    5.roundedRectangleContour parameters.

                   _ r_ulx                       _ r_urx
                  / \                           / \
                 /   \C_ul                A_ur /   \
           B_ul .-----.-----------------------.-----. B_ur     \
               /|                                   |\         |
        r_uly < |                                   | > r_ury  |
               \|                                   |/         |
           A_ul .                                   . C_ur     |
                |                                   |          \
                |                                   |           > h
                |                                   |          /
                |                                   |          |
           A_ll .                                   . C_rl     |
               /|                                   |\         |
        r_lly < |                                   | > r_rly  |
               \|                                   |/         |
           B_ll .-----.-----------------------.-----. B_rl     /
         & (x,y) \   /C_ll                A_rl \   /
                  \_/                           \_/
                    r_llx                         r_rlx
                 \______________  __________________/
                                \/
                                 w

    Note that the ROUNDED_RECT*_NV commands degenerate to the RECT_NV
    command when all the radii are zero.

    PARTIAL ELLIPTICAL ARC COMMAND DETAILS

    In all the arc-based path commands, the parametric segment path
    equations in Table 5.pathEquations are expressed in terms of the
    functions /arc_x/ and /arc_y/.

    Equation 5.generalParametricArc

        arc_x(c,rv,rh,phi,theta1,dtheta,t) = cos(phi)*rh*cos(theta1+t*dtheta) -
                                             sin(phi)*rv*sin(theta1+t*dtheta) + c.x
        arc_y(c,rv,rh,phi,theta1,dtheta,t) = sin(phi)*rh*cos(theta1+t*dtheta) +
                                             cos(phi)*rv*sin(theta1+t*dtheta) + c.y

    This general form of a parametric partial elliptical arc computes
    (x,y) 2D positions on the arc as /t/ ranges from 0.0 to 1.0 inclusive.

    In addition to the varying /t/ parameter, these functions depend on
    a 2D (x,y) center position /c/, a horizontal ellipse radius /rh/,
    a vertical ellipse radius /rv/, a counterclockwise angle (in radians)
    of an ellipse with respect to the x-axis /phi/, /theta1/ is the angle
    (in radians) of the initial point on the partial arc, and /dtheta/
    is the difference between the angle (in radians) of the terminal
    point on the partial arc and /theta1/.  The larger of /rh/ and /rv/
    is the complete ellipse's major axis while the smaller of the two
    is the complete ellipse's minor axis.

    How these additional dependent parameters for /arc_x/ and /arc_y/
    are determined depends on the specific arc path command as
    detailed in Table 5.arcParameterSpecialization.  Before explaining
    how specific arc commands determine these dependent parameters,
    the following discussion develops a general scheme for converting
    general end-point representations of arcs to the partial elliptical
    arc segment representation of Equation 5.generalParametricArc.
    All the arc commands supported are specializations of this general
    end-point representation.  The general scheme is developed, specific
    arc commands are specified as special cases of the general end-point
    representation scheme for arcs.

    In general, consider seven scalar values (/x1/, /y1/, /x2/,
    /y2/, /phi/, /fA/, and /fS/) fully parameterizing a given partial
    elliptical arc:

        *   a 2D position (x1,y1) at the start of a partial elliptical
            arc segment

        *   a 2D position (x2,y2) at the end of a partial elliptical
            arc segment

        *   /phi/ is the angle (in radians) from the x-axis of the path
            space coordinate system to the x-axis of the axis-aligned ellipse

        *   /fA/ is a boolean (the "large arc" flag) that is true when
            the arc spans greater than 180 degrees; and otherwise false
            if the arc sweeps 180 degrees or less

        *   /fS/ is a boolean (the "sweep" flag) that is true when the
            arc sweeps in a counterclockwise direction in path space
            (so sweeps with increasing angles); and otherwise false
            when the arc sweeps in a clockwise direction (so sweeps with
            decreasing angles)

    Given this parameterization, the procedure below computes the /c/,
    /rv/, /rh/, /phi/, /theta1/, and /dtheta/ parameters to represent
    this same arc in the general parametric form of Equation
    5.generalParametricArc.

    Step 1:

       x1p =  cos(phi)*(x1-x2)/2 + sin(phi)*(y1-y2)/2
       y1p = -sin(phi)*(x1-x2)/2 + cos(phi)*(y1-y2)/2

    If /rh/, /rv/, and /phi/ are such that there is no solution
    (basically, the ellipse is not big enough to reach from (x1,y1)
    to (x2,y2), then the ellipse is scaled up uniformly until there
    is exactly one solution (until the ellipse is just big enough)
    in this manner:

       lambda = (x1p/rh)^2 + (y1p/rv)^2

               / rh,               lambda<=1
       rp.x = {
               \ rh*sqrt(lambda),  lambda>1

               / rv,               lambda<=1
       rp.y = {
               \ rv*sqrt(lambda),  lambda>1

    Step 2:

       cp.x = fsgn*sqrt((rp.x^2*rp.y^2 - rp.x^2*y1p^2 - rp.y^2*x1p^2) /
                        (rp.x^2*y1p^2 + rp.y^2*x1p^2)
                       ) * rp.x*y1p/rp.y
       cp.y = fsgn*sqrt((rp.x^2*rp.y^2 - rp.x^2*y1p^2 - rp.y^2*x1p^2) /
                        (rp.x^2*y1p^2 + rp.y^2*x1p^2)
                       ) * -rp.y*x1p/rp.x

    where

                / +1,  fA != fS
        fsgn = {
                \ -1,  fA = fS

    Step 3:

        c.x = cos(phi)*cp.x - sin(phi)*cyp + (x1+x2)/2
        c.y = sin(phi)*cp.x + cos(phi)*cyp + (y1+y2)/2

    In general, the angle between two vectors (u.x, u.y) and (v.x, v.y)
    can be computed as

                      / arcos(dot(u,v)/sqrt(dot(u,u))*sqrt(dot(v,v))),   u.x*v.y-u.y*v.x>=0
        angle(u,v) = {
                      \ -arcos(dot(u,v)/sqrt(dot(u,u))*sqrt(dot(v,v))),  u.x*v.y-u.y*v.x<0

    Step 4:

        theta1 = angle([1,0],
                       [(x1p-cp.x)/r.x,(y1p-cp.y)/r.y])
        dangle = angle([(x1p-cp.x)/r.x,(y1p-cp.y)/r.y],
                       [(-x1p-cp.x)/r.x,(-y1p-cp.y)/r.y])

                    / dangle - 2*Pi,  fS=false AND d>0
                   /
                  /   dangle,         fS=false AND d<=0
        dtheta = {
                  \   dangle,         fS=true  AND d>=0
                   \
                    \ dangle + 2*Pi,  fS=true  AND d<0

    The arc path commands allow arbitrary numeric values so when these
    values result in invalid or out-of-range parameters when the above
    steps are applied, the following further steps are taken to ensure
    well-defined behavior.

    If (x1,y1) and (x2,y2) are identical, then this is equivalent to
    omitting the arc segment entirely.

    If either of /rh/ or /rv/ is zero, the arc is treated as a straight
    line segment from (x1,y1) to (x2,y2).

    Table 5.arcParameterSpecialization now maps the coordinate values
    for each arc path command to the parameters of the arc end-point
    parameterization above from which the arc's parametric representation
    can be obtained.

    Table 5.arcParameterSpecialization: Arc Path Command

    Token                         (x1,y1)     rh         rv         phi          (x2,y2)              fA               fS
    ----------------------------  ----------  ---------  ---------  -----------  -------------------  ---------------  -------
    SMALL_CCW_ARC_TO_NV           cp.x,cp.y   abs(c[0])  abs(c[1])  c[2]*Pi/180  c[3],c[4]            false            true
    RELATIVE_SMALL_CCW_ARC_TO_NV  cp.x,cp.y   abs(c[0])  abs(c[1])  c[2]*Pi/180  cp.x+c[3],cp.y+c[4]  false            true
    SMALL_CW_ARC_TO_NV            cp.x,cp.y   abs(c[0])  abs(c[1])  c[2]*Pi/180  c[3],c[4]            false            false
    RELATIVE_SMALL_CW_ARC_TO_NV   cp.x,cp.y   abs(c[0])  abs(c[1])  c[2]*Pi/180  cp.x+c[3],cp.y+c[4]  false            false
    LARGE_CCW_ARC_TO_NV           cp.x,cp.y   abs(c[0])  abs(c[1])  c[2]*Pi/180  c[3],c[4]            true             true
    RELATIVE_LARGE_CCW_ARC_TO_NV  cp.x,cp.y   abs(c[0])  abs(c[1])  c[2]*Pi/180  cp.x+c[3],cp.y+c[4]  true             true
    LARGE_CW_ARC_TO_NV            cp.x,cp.y   abs(c[0])  abs(c[1])  c[2]*Pi/180  c[3],c[4]            true             false
    RELATIVE_SMALL_CW_ARC_TO_NV   cp.x,cp.y   abs(c[0])  abs(c[1])  c[2]*Pi/180  cp.x+c[3],cp.y+c[4]  true             false
    CIRCULAR_CCW_ARC_TO_NV        A.x,A.y     abs(c[2])  abs(c[2])  0            B.x,B.y              (c[4]-c[3])>180  true
    CIRCULAR_CW_ARC_TO_NV         A.x,A.y     abs(c[2])  abs(c[2])  0            B.x,B.y              (c[4]-c[3])>180  false
    CIRCULAR_TANGENT_ARC_TO_NV    C.x,C.y     abs(c[4])  abs(c[4])  0            D.x,D.y              false            num>=0
    ARC_TO_NV                     cp.x,cp.y   abs(c[0])  abs(c[1])  c[2]*Pi/180  c[5],c[6]            c[3]!=0          c[4]!=0
    RELATIVE_ARC_TO_NV            cp.x,cp.y   abs(c[0])  abs(c[1])  c[2]*Pi/180  cp.x+c[5],cp.y+c[6]  c[3]!=0          c[4]!=0

    where, for CIRCULAR_CCW_ARC_TO_NV and CIRCULAR_CW_ARC_TO_NV,

        A = (c[0]+c[2]*cos(c[3]*Pi/180),
             c[1]+c[2]*sin(c[3]*Pi/180))

        B = (c[0]+c[2]*cos(c[4]*Pi/180),
             c[1]+c[2]*sin(c[4]*Pi/180))

    and C, D, and num, for CIRCULAR_TANGENT_ARC_TO_NV, are computed
    through the following steps:

    Step 1:  Compute two tangent vectors:

        d0.x = cp.x - c[0]
        d0.y = cp.y - c[1]
        d2.x = c[2] - c[0]
        d2.y = c[3] - c[1]

    Step 2:  Compute scaling factors for tangent vectors:

        num   = d0.y*d2.x - d2.y*d0.x
        denom = sqrt(dot(d0,d0)*dot(d2,d2)) - dot(d0,d2)

        dist = abs(c[4] * num/denom)

        l0 = dist/sqrt(dot(d0,d0)) * c[4]/abs(c[4])
        l2 = dist/sqrt(dot(d2,d2)) * c[4]/abs(c[4])

    Step 3:  Add scaled directions to the tangent vector intersection
    point:

             / (c[0],c[1]) + d0 * l0,  denom!=0 AND c[4]!=0
        C = {
             \ (c[0],c[1]),            denom==0 OR c[4]==0

             / (c[0],c[1]) + d2 * l2,  denom!=0 AND c[4]!=0
        D = {
             \ (c[0],c[1]),            denom==0 OR c[4]==0

    PATH OBJECT SPECIFICATION

    Path objects can be specified in one of four ways:

    1)  explicitly from an array of commands and corresponding
        coordinates,

    2)  from a string conforming to one of two supported grammars to
        specify a string,

    3)  from a glyph within a font face from a system font or font file,
        or

    4)  by linearly combining one or more existing path objects with
        mutually consistent command sequences to form a new path.

    In any situation where a path object is specified or re-specified,
    the command's parameters are re-initialized as discussed in section
    5.X.1.5 unless otherwise specified.  However modification of path
    commands and coordinates (section 5.X.1.4) does not modify path
    parameters.

    5.X.1.1 Explicit Path Specification

    The command

        void PathCommandsNV(uint path,
                            sizei numCommands, const ubyte *commands,
                            sizei numCoords, enum coordType,
                            const void *coords);

    specifies a new path object named /path/ where /numCommands/
    indicates the number of path commands, read from the array
    /commands/, with which to initialize that path's command sequence.
    These path commands reference coordinates read sequentially from the
    /coords/ array.  The type of the coordinates read from the /coords/
    array is determined by the /coordType/ parameter which must be
    one of BYTE, UNSIGNED_BYTE, SHORT, UNSIGNED_SHORT, or FLOAT,
    otherwise the INVALID_ENUM error is generated.

    The /numCommands/ elements of the /commands/ array must be tokens
    or character in Table 5.pathCommands.  The command sequence matches
    the element order of the /commands/ array.  Each command references
    a number of coordinates specified by "Coordinate count" column of
    Table 5.pathCommands, starting with the first (zero) element of
    the /coords/ array and advancing by the coordinate count for each
    command.  If any of these /numCommands/ command values are not
    listed in the "Token" or "Character aliases" columns of Table
    5.pathCommands, the INVALID_ENUM error is generated.

    The INVALID_OPERATION error is generated if /numCoords/ does not
    equal the number of coordinates referenced by the command sequence
    specified by /numCommands/ and /commands/ (so /numCoords/ provides a
    sanity check that the /coords/ array is being interpreted properly).
    The error INVALID_VALUE is generated if either /numCommands/ or
    /numCoords/ is negative.

    If the PathCommandsNV command results in an error, the path object
    named /path/ is not changed; if there is no error, the prior contents
    of /path/, if /path/ was an existent path object, are lost and the
    path object name /path/ becomes used.

    5.X.1.2 String Path Specification

    The command

        void PathStringNV(uint path, enum format,
                          sizei length, const void *pathString);

    specifies a new path object named /path/ where /format/ must be
    either PATH_FORMAT_SVG_NV or PATH_FORMAT_PS_NV, in which case the
    /length/ and /pathString/ are interpreted according to grammars
    specified in sections 5.X.1.2.1 and 5.X.1.2.2 respectively.
    The INVALID_VALUE error is generated if /length/ is negative.

    If the PathStringNV command results in an error, the path object
    named /path/ is not changed; if there is no error, the prior contents
    of /path/, if /path/ was an existent path object, are lost and the
    path object name /path/ becomes used.

    5.X.1.2.1 Scalable Vector Graphics Path Grammar

    If the /format/ parameter of PathStringNV is PATH_FORMAT_SVG_NV,
    the /pathString/ parameter is interpreted as a string of ubyte ASCII
    characters with /length/ elements.

    This string must satisfy the "svg-path" production in the path
    grammar below.  This grammar is taken directly from the Scalable
    Vector Graphics (SVG) 1.1 (April 30, 2009) specification.

    The following notation is used in the Backus-Naur Form (BNF)
    description of the grammar for an SVG path string:

        * *: 0 or more
        * +: 1 or more
        * ?: 0 or 1
        * (): grouping
        * ()^n: grouping with n repetitions where n is explained subsequently
        * |: separates alternatives
        * double quotes surround literals
        * #x: prefixes an ASCII character value followed by hexadecimal
          digits
        * ..: means any of an inclusive range of ASCII characters, so
          '0'..'9' means any digit character

    The following is the grammar for SVG paths.

        svg-path:
            wsp* moveto-drawto-command-groups? wsp*
        moveto-drawto-command-groups:
            moveto-drawto-command-group
            | moveto-drawto-command-group wsp* moveto-drawto-command-groups
        moveto-drawto-command-group:
            moveto wsp* drawto-commands?
        drawto-commands:
            drawto-command
            | drawto-command wsp* drawto-commands
        drawto-command:
            closepath
            | lineto
            | horizontal-lineto
            | vertical-lineto
            | curveto
            | smooth-curveto
            | quadratic-bezier-curveto
            | smooth-quadratic-bezier-curveto
            | elliptical-arc
        moveto:
            ( "M" | "m" ) wsp* moveto-argument-sequence
        moveto-argument-sequence:
            coordinate-pair
            | coordinate-pair comma-wsp? lineto-argument-sequence
        closepath:
            ("Z" | "z")
        lineto:
            ( "L" | "l" ) wsp* lineto-argument-sequence
        lineto-argument-sequence:
            coordinate-pair
            | coordinate-pair comma-wsp? lineto-argument-sequence
        horizontal-lineto:
            ( "H" | "h" ) wsp* horizontal-lineto-argument-sequence
        horizontal-lineto-argument-sequence:
            coordinate
            | coordinate comma-wsp? horizontal-lineto-argument-sequence
        vertical-lineto:
            ( "V" | "v" ) wsp* vertical-lineto-argument-sequence
        vertical-lineto-argument-sequence:
            coordinate
            | coordinate comma-wsp? vertical-lineto-argument-sequence
        curveto:
            ( "C" | "c" ) wsp* curveto-argument-sequence
        curveto-argument-sequence:
            curveto-argument
            | curveto-argument comma-wsp? curveto-argument-sequence
        curveto-argument:
            coordinate-pair comma-wsp? coordinate-pair comma-wsp? coordinate-pair
        smooth-curveto:
            ( "S" | "s" ) wsp* smooth-curveto-argument-sequence
        smooth-curveto-argument-sequence:
            smooth-curveto-argument
            | smooth-curveto-argument comma-wsp? smooth-curveto-argument-sequence
        smooth-curveto-argument:
            coordinate-pair comma-wsp? coordinate-pair
        quadratic-bezier-curveto:
            ( "Q" | "q" ) wsp* quadratic-bezier-curveto-argument-sequence
        quadratic-bezier-curveto-argument-sequence:
            quadratic-bezier-curveto-argument
            | quadratic-bezier-curveto-argument comma-wsp?
                quadratic-bezier-curveto-argument-sequence
        quadratic-bezier-curveto-argument:
            coordinate-pair comma-wsp? coordinate-pair
        smooth-quadratic-bezier-curveto:
            ( "T" | "t" ) wsp* smooth-quadratic-bezier-curveto-argument-sequence
        smooth-quadratic-bezier-curveto-argument-sequence:
            coordinate-pair
            | coordinate-pair comma-wsp? smooth-quadratic-bezier-curveto-argument-sequence
        elliptical-arc:
            ( "A" | "a" ) wsp* elliptical-arc-argument-sequence
        elliptical-arc-argument-sequence:
            elliptical-arc-argument
            | elliptical-arc-argument comma-wsp? elliptical-arc-argument-sequence
        elliptical-arc-argument:
            nonnegative-number comma-wsp? nonnegative-number comma-wsp?
                number comma-wsp flag comma-wsp flag comma-wsp coordinate-pair
        coordinate-pair:
            coordinate comma-wsp? coordinate
        coordinate:
            number
        nonnegative-number:
            integer-constant
            | floating-point-constant
        number:
            sign? integer-constant
            | sign? floating-point-constant
        flag:
            "0" | "1"
        comma-wsp:
            (wsp+ comma? wsp*) | (comma wsp*)
        comma:
            ","
        integer-constant:
            digit-sequence
        floating-point-constant:
            fractional-constant exponent?
            | digit-sequence exponent
        fractional-constant:
            digit-sequence? "." digit-sequence
            | digit-sequence "."
        exponent:
            ( "e" | "E" ) sign? digit-sequence
        sign:
            "+" | "-"
        digit-sequence:
            digit
            | digit digit-sequence
        digit:
            "0".."9"
        wsp:
            (#x20 | #x9 | #xD | #xA)

    The processing of the BNF must consume as much of a given BNF
    production as possible, stopping at the point when a character
    is encountered which no longer satisfies the production.  Thus,
    in the string "M 100-200", the first coordinate for the "moveto"
    consumes the characters "100" and stops upon encountering the minus
    sign because the minus sign cannot follow a digit in the production
    of a "coordinate".  The result is that the first coordinate will be
    "100" and the second coordinate will be "-200".

    Similarly, for the string "M 0.6.5", the first coordinate of the
    "moveto" consumes the characters "0.6" and stops upon encountering
    the second decimal point because the production of a "coordinate"
    only allows one decimal point. The result is that the first coordinate
    will be "0.6" and the second coordinate will be ".5".

    The grammar allows the string to be empty (zero length).  This is
    not an error, instead specifies a path with no commands.

    Table 5.svgCommands maps productions in the grammar above to the
    path commands in Table 5.pathCommands; each such path command, with
    its corresponding coordinates, is added to the path command sequence
    of the path object.  Each production listed in Table 5.svgCommands
    consumes a number of coordinates consistent with the path command
    token's coordinate count listed in Table 5.pathCommands.  The
    "coordinate" and "nonnegative-number" productions convert to a numeric
    coordinate value in the obvious way.  The "flag" production converts
    "0" and "1" to numeric coordinate values zero and one respectively.

    Table 5.svgCommands: SVG Grammar Commands to Path Command Tokens

                                                           Grammar's prior
        Production                                         command character  Path command token
        -------------------------------------------------  -----------------  -------------------------------------
        moveto-argument-sequence                           "M"                MOVE_TO_NV
                                                           "m"                RELATIVE_MOVE_TO_NV
        closepath                                          "Z" or "z"         CLOSE_PATH_NV
        lineto-argument-sequence                           "L"                LINE_TO_NV
                                                           "l"                RELATIVE_LINE_TO_NV
        horizontal-lineto-argument-sequence                "H"                HORIZONTAL_LINE_TO_NV
                                                           "h"                RELATIVE_HORIZONTAL_LINE_TO_NV
        vertical-lineto-argument-sequence                  "V"                VERTICAL_LINE_TO_NV
                                                           "v"                RELATIVE_VERTICAL_LINE_TO_NV
        quadratic-bezier-curveto-argument                  "Q"                QUADRATIC_CURVE_TO_NV
                                                           "q"                RELATIVE_QUADRATIC_CURVE_TO_NV
        smooth-quadratic-bezier-curveto-argument-sequence  "T"                SMOOTH_QUADRATIC_CURVE_TO_NV
                                                           "t"                RELATIVE_SMOOTH_QUADRATIC_CURVE_TO_NV
        curveto-argument                                   "C"                CUBIC_CURVE_TO_NV
                                                           "c"                RELATIVE_CUBIC_CURVE_TO_NV
        smooth-curveto-argument                            "S"                SMOOTH_CUBIC_CURVE_TO_NV
                                                           "s"                RELATIVE_SMOOTH_CUBIC_CURVE_TO_NV
        elliptical-arc-argument                            "A"                ARC_TO_NV
                                                           "a"                RELATIVE_ARC_TO_NV

    If the string fails to satisfy the svg-path production, the path
    object named /path/ is not changed.  The production may not be
    satisfied for one of two reasons: either the grammar cannot be not
    satisfied by the string, or the grammar is satisfied but there still
    remain a non-zero number of characters in the string.  Neither
    failure to satisfy the production generates an error; instead the
    PATH_ERROR_POSITION_NV state is set to the character offset where the
    grammar was first not satisfied or where the grammar was exhausted.
    If the string was parsed successfully and the command did not generate
    an error, the PATH_ERROR_POSITION_NV state is set to negative one
    to indicate success.

    5.X.1.2.2 PostScript Path Grammar

    If the /format/ parameter of PathStringNV is PATH_FORMAT_PS_NV,
    the /pathString/ parameter is interpreted as a string of ubyte ASCII
    characters with /length/ elements.

    This string must satisfy the "ps-path" production in the path
    grammar below.  This grammar is parses path specified in PostScript's
    subgrammar for user paths specified by "PostScript Language Reference
    Manual" 3rd edition.

    The following is the grammar (using the same notation as section
    5.X.1.2.1) for PS paths with special support for binary encoding modes
    (as explained below):

        ps-path:
            ps-wsp* user-path? ps-wsp*
            | ps-wsp* encoded-path ps-wsp*
        user-path:
            user-path-cmd
            | user-path-cmd ps-wsp+ user-path
        user-path-cmd:
            setbbox
            | ps-moveto
            | rmoveto
            | ps-lineto
            | rlineto
            | ps-curveto
            | rcurveto
            | arc
            | arcn
            | arct
            | ps-closepath
            | ucache
        setbbox:
            numeric-value numeric-value numeric-value numeric-value setbbox-cmd
        setbbox-cmd:
            "setbbox"
            | #x92 #x8F
        ps-moveto:
            numeric-value numeric-value moveto-cmd
        moveto-cmd:
            "moveto"
            | #x92 #x6B
        rmoveto:
            numeric-value numeric-value rmoveto-cmd
        rmoveto-cmd:
            "rmoveto"
            | #x92 #x86
        ps-lineto:
            numeric-value numeric-value lineto-cmd
        lineto-cmd:
            "lineto"
            | #x92 #x63
        rlineto:
            numeric-value numeric-value rlineto-cmd
        rlineto-cmd:
            "rlineto"
            | #x92 #x85
        ps-curveto:
            numeric-value numeric-value numeric-value numeric-value numeric-value numeric-value curveto-cmd
        curveto-cmd:
            "curveto"
            | #x92 #x2B
        rcurveto:
            numeric-value numeric-value numeric-value numeric-value numeric-value numeric-value rcurveto-cmd
        rcurveto-cmd:
            "rcurveto"
            | #x92 #x7A
        arc:
            numeric-value numeric-value numeric-value numeric-value numeric-value arc-cmd
        arc-cmd:
            "arc"
            | #x92 #x05
        arcn:
            numeric-value numeric-value numeric-value numeric-value numeric-value arcn-cmd
        arcn-cmd:
            "arcn"
            | #x92 #x06
        arct:
            numeric-value numeric-value numeric-value numeric-value numeric-value arct-cmd
        arct-cmd:
            "arct"
            | #x92 #x07
        ps-closepath:
            "closepath"
            | #x92 #x16
        ucache:
            "ucache"
            | #x92 #xB1
        encoded-path:
            data-array ps-wsp* operator-string
        data-array:
            "{" ps-wsp* numeric-value-sequence? "}"
            | homogeneous-number-array
            | ascii85-homogeneous-number-array
        operator-string:
            hexadecimal-binary-string
            | ascii85-string
            | short-binary-string
            | be-long-binary-string
            | le-long-binary-string
        hexadecimal-binary-string:
            "<" ps-wsp-chars* hexadecimal-sequence ps-wsp-chars* ">"
        hexadecimal-sequence:
            hexadecimal-digit
            | hexadecimal-digit ps-wsp-chars* hexadecimal-sequence
        hexadecimal-digit:
            digit
            | "a".."f" |
            | "A".."F"
        short-binary-string:
            #x8E one-byte ( one-byte )^n
                /where n is the value of the one-byte production decoded
                 as an unsigned integer, 0 through 255/
        be-long-binary-string:
            #x8F two-bytes ( one-byte )^n
                /where n is the value of the two-bytes production decoded
                 as an unsigned integer, 0 through 65535, decoded in
                 big-endian byte order/
        le-long-binary-string:
            #x90 two-bytes ( one-byte )^n
                /where n is the value of the two-bytes production decoded
                 as an unsigned integer, 0 through 65535, decoded in
                 little-endian byte order/
        numeric-value-sequence:
            numeric-value:
            | numeric-value numeric-value-sequence
        numeric-value:
            number ps-wsp+
            | radix-number ps-wsp+
            | be-integer-32bit
            | le-integer-32bit
            | be-integer-16bit
            | le-integer-16bit
            | le-integer-8bit
            | be-fixed-16bit
            | le-fixed-16bit
            | be-fixed-32bit
            | le-fixed-32bit
            | be-float-ieee
            | le-float-ieee
            | native-float-ieee
        be-integer-32bit:
            #x84 four-bytes
        le-integer-32bit:
            #x85 four-bytes
        be-integer-16bit:
            #x86 two-bytes
        le-integer-16bit:
            #x87 two-bytes
        le-integer-8bit:
            #x88 one-byte
        be-fixed-32bit:
            #x89 #x0..#x1F four-bytes
        le-fixed-32bit:
            #x89 #x80..#x9F four-bytes
        be-fixed-16bit:
            #x89 #x20..#x2F two-bytes
        le-fixed-16bit:
            #x89 #xA0..#xAF two-bytes
        be-float-ieee:
            #x8A four-bytes
        le-float-ieee:
            #x8B four-bytes
        native-float-ieee:
            #x8C four-bytes
        radix-number:
            base "#" base-number
        base:
            digit-sequence
        base-number:
            base-digit-sequence
        base-digit-sequence:
            base-digit
            | base-digit base-digit-sequence
        base-digit:
            digit
            | "a".."z"
            | "A".."Z"
        homogeneous-number-array:
            be-fixed-32bit-array
            | be-fixed-16bit-array
            | be-float-ieee-array
            | native-float-ieee-array
            | le-fixed-32bit-array
            | le-fixed-16bit-array
            | le-float-ieee-array
        be-fixed-32bit-array:
            #x95 #x0..#x1F two-bytes ( four-bytes )^n
                /where n is the value of the two-bytes production decoded
                 as an unsigned integer, 0 through 65535, decoded in
                 big-endian byte order/
        be-fixed-16bit-array:
            #x95 #x20..#x2F two-bytes ( two-bytes )^n
                /where n is the value of the two-bytes production decoded
                 as an unsigned integer, 0 through 65535, decoded in
                 big-endian byte order/
        be-float-ieee-array:
            #x95 #x30 two-bytes ( four-bytes )^n
                /where n is the value of the two-bytes production decoded
                 as an unsigned integer, 0 through 65535, decoded in
                 big-endian byte order/
        le-fixed-32bit-array:
            #x95 #x80..#x9F two-bytes ( four-bytes )^n
                /where n is the value of the two-bytes production decoded
                 as an unsigned integer, 0 through 65535, decoded in
                 little-endian byte order/
        le-fixed-16bit-array:
            #x95 #xA0..#xAF two-bytes ( two-bytes )^n
                /where n is the value of the two-bytes production decoded
                 as an unsigned integer, 0 through 65535, decoded in
                 little-endian byte order/
        le-float-ieee-array:
            #x95 #xB0 two-bytes ( four-bytes )^n
                /where n is the value of the two-bytes production decoded
                 as an unsigned integer, 0 through 65535, decoded in
                 little-endian byte order/
        native-float-ieee-array:
            #x95 ( #x31 | #xB1 ) two-bytes ( four-bytes )^n
                /where n is the value of the two-bytes production decoded
                 as an unsigned integer, 0 through 65535, decoded in
                 the native byte order/
        ascii85-string:
            "<~" (#x21..#x75 | "z" | psp-wsp )* "~>"
        ascii85-homogeneous-number-array:
            "<~" (#x21..#x75 | "z" | psp-wsp )* "~>"
        one-byte:
            #x0..#xFF
        two-bytes:
            #x0..#xFF #x0..#xFF
        four-bytes:
            #x0..#xFF #x0..#xFF #x0..#xFF #x0..#xFF
        ps-wsp:
            ps-wsp-chars
            | ps-comment
        ps-wsp-chars:
            ( #x20 | #x9 | #xA | #xC | #xD | #x0 )
        ps-comment:
            "%" ( #0..#9 | #xB..#xC | #xE..#xFF )* ( #xD | #xA )

    This grammar is not technically a pure BNF because it uses binary
    encoded data to encode how many characters should be as part of
    several productions (short-binary-string, native-float-ieee-array,
    etc.).

    The processing of the BNF must consume as much of a given BNF
    production as possible, stopping at the point when a character
    is encountered which no longer satisfies the production.

    The grammar allows the string to be empty (zero length). This
    is not an error, instead specifies a path with no commands.

    Table 5.psCommands maps productions in the grammar above to the path
    commands in Table 5.pathCommands; each such path command, with its
    corresponding coordinates, is added to the path command sequence
    of the path object.  Each production listed in Table 5.svgCommands
    consumes a quantity of values, matched by the "number" production,
    consistent with the path command token's coordinate count listed
    in Table 5.pathCommands.  The "setbbox" and "ucache" products are
    matched but do not result in path commands.

    Table 5.psCommands: PS Grammar Commands to Path Command Tokens

        Production    Path command token
        ------------  --------------------------
        arc           CIRCULAR_CCW_ARC_TO_NV
        arcn          CIRCULAR_CW_ARC_TO_NV
        arct          CIRCULAR_TANGENT_ARC_TO_NV
        ps-closepath  CLOSE_PATH_NV
        ps-curveto    CUBIC_CURVE_TO_NV
        ps-lineto     LINE_TO_NV
        ps-moveto     MOVE_TO_NV
        rcurveto      RELATIVE_CUBIC_CURVE_TO_NV
        rlineto       RELATIVE_LINE_TO_NV
        rmoveto       RELATIVE_MOVE_TO_NV
        setbbox       -
        ucache        -

    The "number" production converts to a numeric coordinate value
    in the obvious way.  The "radix-number" production converts the
    base-n integer conversion of its "base-number" production using
    the base indicated by the base-10 integer conversion of its "base"
    production where the base /n/ must be within the range 2 to 26.
    The "base-number" is interpreted in base /n/; the "base-number"
    production must contain digits ranging from 0 to /n/-1; digits greater
    than 9 are represented by the letters A through Z (or a through z)
    for the values 10 through 35 respectively.

    The "encoded-path" production provides a compact and precise way
    to encode paths with the commands and coordinates decoupled.

    The "data-array" subproductions provide a sequence of coordinate
    values for the encoded path's commands.  The "data-array"
    subproduction provides a sequence of numbers that is used by the
    following "operator-string" production.

    The "operator-string" subproduction is interpreted as a sequence
    of encoded path commands, one command per byte generated by
    "operator-string"'s "binary-string" production.

    Each hexadecimal character in the "hexadecimal-binary-string"
    production is a nibble (a 4-bit quantity).  Each pair of characters
    is two nibbles and they form a byte with the first nibble
    representing the most signification bits of the byte.  If the
    "hexadecimal-binary-string" production contains an odd number of
    hexadecimal characters, "0" is assumed to be suffixed to make an
    even number of characters (so "A7C" would encode the bytes 167 for
    "A7" followed by 192 for "C" which is treated as "C0" for 192).
    Table 5.encodedPathOpcodes maps the values contained in the operator
    string to path commands.  Each command consumes from the coordinate
    array supplied by the "data-array" production a number of values
    for the command's coordinates equal to the path command token's
    coordinate count listed in Table 5.pathCommands.  If the value for
    an element of the operator string is between 12 and 32 inclusive,
    the grammar fails to parse at this point.  If the value /n/ of an
    element of the operator string is between 32 and 255, then this value
    /n/-32 is treated as a repetition count and is treated as if /n/-32
    repetitions of the next command are contained in the operator string
    instead and the appropriate number of coordinates are consumed from
    the associated sequence of coordinate values.

    Table 5.encodedPathOpcodes

        Opcode  Name
        ------  ---------
        0       setbbox
        1       moveto
        2       rmoveto
        3       lineto
        4       rlineto
        5       curveto
        6       rcurveto
        7       arc
        8       arcn
        9       arct
        10      closepath
        11      ucache

    The ASCII characters in the "ascii85-binary-string" production
    consists of a sequence of printable ASCII characters between the "<~"
    and "~>" delimiters.  This represents arbitrary binary data using
    an encoding technique that products a 4:5 expansion as opposed to
    the 1:2 expansion for the "hexadecimal-binary-string" production.
    This encoding is known as ASCII base-85.

    Binary data in the ASCII base-85 encoding are encoded in 4-tuples
    (groups of 4) each 4-tuple is used to produce a 5-type of ASCII
    characters.  If the binary 4-tuple is (b1,b2,b3,b4) and the encoded
    5-tuple is (c1,c2,c3,c4,c5), then the relation between them is:

       (b1 * 256^3) + (b2 * 256^2) + (b3 * 256^1) + b4 =
       (c1 * 256^4) + (c2 * 256^3) + (c3 * 256^2) + (c4 * 256^3) + c5

    The four bytes of binary data are interpreted as a base-256 number and
    then converted into a base-85 number.  The five "digits" of this number,
    (c1,c2,c3,c4,c5), are then converted into ASCII characters by adding 33,
    which is the ASCII code for '!', to each.  ASCII characters in the
    range '!' to 'u' are used, where '!' represented the value 0 and 'u'
    represents the value 84.  As a special case, if all five digits are
    zero, they must be represented by either a single 'z' instead of by
    '!!!!'.

    If the encoded sequence ends with a sequence of characters that is
    not an even multiple of 4, the last 1, 2, or 3 characters to produce
    a special final partial 5-tuple.  Given n (1, 2, or 3) bytes of final
    binary data, an encoder must first append 4-n zero bytes to make
    a complete 4-tuple.  Then, the encoder must encode the 4-tuple in
    the usual way, but without applying the 'z' special case.  Finally,
    the encoder must write the first n+1 bytes of the resulting 5-tuple.
    Those bytes are immediately followed by the "~>" terminal marker.

    This encoding scheme is reversible and the GL is responsible for
    converting the ASCII base-85 string into its corresponding binary
    data.  White space within an ASCII base-85 encoded string is ignored.

    The following conditions constitute encoding violations of the ASCII
    base-85 scheme:

        *   The value represented by a 5-tuple is greater than 2^32-1

        *   The 'z' value occurs in the middle of a 5-tuple.

        *   A final partial 5-tuple contains only one character.

    Any such encoding violation is a parsing error.

    Once the ASCII base-85 string is decoded, this sequence of bytes
    is treated as operator elements in the identical manner as the
    elements for the "hexadecimal-string" subproduction.  This means
    invalid opcodes are possible and are treated as parsing errors, and
    Valid opcodes and counts consume coordinates from the "data-array"
    production to generate path commands with associated coordinates.

    The "short-binary-string", "be-long-binary-string", and
    "le-long-binary-string" subproductions of "operator-string" are
    binary encodings of a sequence of operator string elements.

    The "short-binary-string" has a count from 0 to 255 supplied by its
    "one-byte" subproduction which indicates how many bytes follow.
    These remaining (unsigned) bytes generate the sequence of operator
    string elements.

    The "be-long-binary-string" has a count from 0 to 65535 supplied by
    its "two-byte" subproduction which indicates how many bytes follow.
    These remaining (unsigned) bytes generate the sequence of operator
    string elements.  The "two-byte" subproduction is converted to a
    count by multiplying the first unsigned byte by 256 and adding it
    to the second unsigned byte.

    The "le-long-binary-string" has a count from 0 to 65535 supplied by
    its "two-byte" subproduction which indicates how many bytes follow.
    These remaining (unsigned) bytes generate the sequence of operator
    string elements.  The "two-byte" subproduction is converted to a
    count by multiplying the second unsigned byte by 256 and adding it
    to the first unsigned byte.

    The "encoded-path" fails to parse if invalid opcodes are detected
    in the operator string or the sequence of numbers for coordinates
    is exhausted prematurely.

    If the string fails to satisfy the ps-path production, the path
    object named /path/ is not changed.  The production may not be
    satisfied for one of three reasons: the grammar cannot be not
    satisfied by the string, the string has invalid sequences (such
    as ASCII base-85 violations, exhausting the coordinate data in the
    "data-array" production, or invalid opcodes encountered in the
    "operator-string" production), or the grammar is satisfied but
    there still remain a non-zero number of characters in the string.
    None of these failures to satisfy the grammar generates an error;
    instead the PATH_ERROR_POSITION_NV state is set to the character
    offset where the grammar was first not satisfied, violated
    semantically, or where the grammar was exhausted.  If the string
    was parsed successfully and the command did not generate an error,
    the PATH_ERROR_POSITION_NV state is set to negative one to indicate
    success.

    If a parsing error occurs, the exact value assigned to the
    PATH_ERROR_POSITION_NV state variable is implementation-dependent
    (because the specifics of error position determination is difficult
    to specify) though the determined error location should be nearby
    the first error.

    5.X.1.3 Font Glyph Path Specification

    PATH GLYPHS FROM CHARACTER CODE SEQUENCE

    The command

        void PathGlyphsNV(uint firstPathName,
                          enum fontTarget,
                          const void *fontName,
                          bitfield fontStyle,
                          sizei numGlyphs, enum type,
                          const void *charcodes,
                          enum handleMissingGlyphs,
                          uint pathParameterTemplate,
                          float emScale);

    creates, if no error occurs, a range of path objects named from
    /firstPathName/ to /firstPathName/+/numGlyphs/-1 based on the
    font face indicated by /fontTarget/, /fontName/, and /fontStyle/
    and the sequence of /numGlyphs/ character codes listed in the
    /charcodes/ array, as interpreted based by the /type/ parameter.
    However each particular name in the range /firstPathName/ to
    /firstPathName/+/numGlyphs/-1 is specified as a new path object only
    if that name is not already in use as a path object; if a name is
    already in use, that named path object is silently left undisturbed.
    A path object name is also left undisturbed if the
    /handleMissingGlyphs/ parameter is SKIP_MISSING_GLYPH_NV and the
    character code for a given glyph corresponds to the font's missing
    glyph or the character code is otherwise not available.

    The error INVALID_VALUE is generated if /numGlyphs/ or /emScale/
    is negative.

    The /fontTarget/ parameter must be one of STANDARD_FONT_NAME_NV,
    SYSTEM_FONT_NAME_NV, or FILE_NAME_NV; otherwise the INVALID_ENUM
    error is generated.

    The /handleMissingGlyphs/ parameter must be one of
    SKIP_MISSING_GLYPH_NV or USE_MISSING_GLYPH_NV; otherwise the
    INVALID_ENUM error is generated.

    If /fontTarget/ is STANDARD_FONT_NAME_NV, then /fontName/ is
    interpreted as a nul-terminated 8-bit ASCII character string that
    must be one of the following strings: "Serif", "Sans", "Mono",
    or "Missing"; otherwise the INVALID_VALUE error is generated.
    These "Serif", "Sans", and "Mono" names respectively correspond to
    serif, sans-serif, and sans monospaced font faces with the intent
    that the font face matches the appearance, metrics, and kerning
    of the DejaVu fonts of the same names.  All implementations /must/
    support these font names for the STANDARD_FONT_NAME_NV target.

    For the STANDARD_FONT_NAME_NV targets with "Serif", "Sans", and
    "Mono", all implementations /must/ support the first 256 character
    codes defined by Unicode and the ISO/IEC 8859-1 (Latin-1 Western
    European) character encoding though implementations are strongly
    encouraged to support as much of the Unicode character codes as the
    system's underlying font and language support provides.

    For the STANDARD_FONT_NAME_NV targets with "Missing", the entire
    sequence of path objects must be populated with an identical box
    outline with metrics matching this box.

    If /fontTarget/ is SYSTEM_FONT_NAME_NV, then /fontName/ is interpreted
    as a nul-terminated 8-bit ASCII character string that corresponds to a
    system-specific font name.  These names are intended to correspond to
    the fonts names typically used in web content (e.g. Arial, Georgia,
    Times Roman, Helvetica).  The mapping of the system font character
    string to a system font is assumed to be performed by the GL server.

    If /fontTarget/ is FILE_NAME_NV, then /fontName/ is interpreted as
    a nul-terminated 8-bit ASCII character string that corresponds to
    a system-specific file name in a standard outline font format.
    The specific interpretation of this name depends on the system
    conventions for identifying files by name.  This name can be an
    absolute or relative path.  The name is expected to include the
    font name's extension.  The mapping of the font file name to a
    font is assumed to be performed by the GL client.  What font file
    formats are supported is system dependent but implementations are
    encouraged to support outline font formats standard to the system
    (e.g. TrueType for Windows systems, etc.).

    If the /fontTarget/ and /fontName/ combination can not be loaded for
    any reason (including the file name could not be opened, the font
    name is not available on the system, the font file format is not
    supported, the font file format is corrupted, etc.) and there is no
    other error generated, the command succeeds silently (so no error
    is generated) and the range of named path objects is not modified.
    If the named path objects did not exist previously, they continue
    to not exist.

    The /fontStyle/ parameter is a bitfield allowed to have the
    bits BOLD_BIT_NV or ITALIC_BIT_NV set; if other bits are set, the
    INVALID_VALUE error is generated.  The font style is used as a hint to
    indicate the style of the font face.  Glyphs are generated with the
    font's bold or italic style respectively (or combination thereof)
    if the BOLD_BIT_NV or ITALIC_BIT_NV bits are set; otherwise, the
    value 0 or NONE indicates the default font face style should be used
    to generate the requested glyphs.  In situations where the bold or
    italic style of the font is encoded in the font name or file name,
    the /fontStyle/ parameter is ignored.

    The generated glyphs for the path objects named /firstPathName/
    to /firstPathName/+/numGlyphs/-1 are specified by the /numGlyphs/
    character codes listed in the /charcodes/ array where each element of
    the array is determined by the /type/ parameter that must be one of
    UNSIGNED_BYTE, UNSIGNED_SHORT, UNSIGNED_INT, UTF8_NV, UTF16_NV,
    2_BYTES, 3_BYTES, and 4_BYTES with the array accessed in the same
    manner as the CallLists command's /type/ and /lists/ parameters
    (though not offset by the display list base), but indicating character
    codes instead of display list names.

    The character codes from the /charcodes/ array are Unicode character
    codes if the font in question can map from the Unicode character
    set to the font's glyphs.  If the font has no meaningful mapping
    from Unicode, the font's standard character set is used instead
    of Unicode (e.g. a font filled with non-standard symbols).  For a
    font supporting a character set that can be mapped to the Unicode
    character set, a best effort should be made to map the specified
    character code from its Unicode character code interpretation to
    the closest appropriate glyph in the specified font.

    Path objects created from glyphs by PathGlyphsNV have their path
    object metric state initialized from the metrics of the glyph from
    which they were specified.  Section 6.X.3. ("Path Object Glyph
    Typographic Queries") explains how these metrics are queried and
    what their values mean.  While the per-glyph metrics are expected to
    vary from glyph to glyph within a font face, the per-font metrics
    are expected to be identical for every path object created from a
    given font name and font style combination.

    Metrics in font space of glyphs are scaled by a value /s/ that is the
    ratio of the /emScale/ parameter divided by the font's units per Em;
    if the /emScale/ parameter equals zero, treat /emScale/ as if it was
    identical to the font's units per Em such that /s/ is exactly 1.0.
    Each glyph's outline are also scaled by /s/.  The metric values /not/
    scaled by /s/ are GLYPH_HAS_KERNING_BIT_NV, FONT_UNITS_PER_EM_BIT_NV,
    FONT_HAS_KERNING_BIT_NV, and FONT_NUM_GLYPH_INDICES_BIT_NV (since
    these metric values are not specified in font units).

    The FONT_NUM_GLYPH_INDICES_BIT_NV metric value returns -1 for path
    objects created with the STANDARD_FONT_NAME_NV (as such fonts are
    not accessed by glyph index, only character point); otherwise, the
    value is number of glyphs indices for the font, whether or not the
    path object is created from a character point or glyph index.

    When unknown or missing character codes in a font face are specified
    and the /handleMissingGlyph/ parameter is USE_MISSING_GLYPHS_NV,
    this situation should be handled in a manner appropriate to the
    character code, font face, and implementation.  Typically this
    involves using the font's missing glyph for the unknown or missing
    character code.

    If the /pathParameterTemplate/ parameter names an existing path
    object, that path object's current parameters listed in Table
    5.pathParameters (excepting PATH_FILL_MODE_NV as explained in
    the following paragraph) are used to initialize the respective
    parameters of path objects specified by this command; otherwise
    if the /pathParameterTemplate/ path object name does not exist,
    the initial path parameters are used as specified by table 6.Y
    (without generating an error).

    Path objects created from glyphs by PathGlyphsNV have their
    PATH_FILL_MODE_NV parameter, as explained in Section 5.X.1.5 ("Path
    Parameter Specification"), initialized according to the fill
    conventions of the font outlines within the font (instead of the
    COUNT_UP_NV default for paths specified by means other than glyphs).
    This may be one of:  COUNT_UP_NV if the font's outline winding
    convention is counterclockwise and its outline filling assumes the
    non-zero winding rule; COUNT_DOWN_NV if the font's outline winding
    convention is clockwise and its outline filling assumes the non-zero
    winding rule; or INVERT if the font's outline filling assumes the
    even-odd winding rule.

    PATH GLYPHS FROM CHARACTER CODE RANGE

    The command

        void PathGlyphRangeNV(uint firstPathName,
                              enum fontTarget,
                              const void *fontName,
                              bitfield fontStyle,
                              uint firstGlyph,
                              sizei numGlyphs,
                              enum handleMissingGlyphs,
                              uint pathParameterTemplate,
                              float emScale);

    allows a sequence of character codes in a font face to specify a
    sequence of path objects and is equivalent to

        int *array = malloc(sizeof(int)*numGlyphs);
        if (array) {
          for (int i=0; i<numGlyphs; i++) {
            array[i] = i + firstGlyph;
          }
          PathGlyphsNV(firstPathName, fontTarget, fontName, fontStyle,
                       numGlyphs, INT, array,
                       handleMissingGlyphs, pathParameterTemplate, emScale);
          free(array);
        } else {
          // generate OUT_OF_MEMORY error
        }

    PATH GLYPHS FROM GLYPH INDEX RANGE

    Advanced shaping of text renders glyphs by per-font glyph indices
    (rather than Unicode code point).  The commands

        enum PathGlyphIndexArrayNV(uint firstPathName,
                                   enum fontTarget,
                                   const void *fontName,
                                   bitfield fontStyle,
                                   uint firstGlyphIndex,
                                   sizei numGlyphs,
                                   uint pathParameterTemplate,
                                   float emScale);

        enum PathMemoryGlyphIndexArrayNV(uint firstPathName,
                                         enum fontTarget,
                                         sizeiptr fontSize,
                                         const void *fontData,
                                         sizei faceIndex,
                                         uint firstGlyphIndex,
                                         sizei numGlyphs,
                                         uint pathParameterTemplate,
                                         float emScale);

    create, if successful and no error occurs, a range of path objects
    that correspond to an array of glyphs as ordered by glyph index in
    a font face.  PathGlyphIndexArrayNV loads the font data from a file
    name or system font name while PathMemoryGlyphIndexArrayNV loads
    the font data from a standard font format in system memory.

    The commands return the value FONT_GLYPHS_AVAILABLE_NV when
    successful; otherwise one of the following values is returned
    depending on the nature of the failure.  The unsuccessful command
    returns the value FONT_TARGET_UNAVAILABLE_NV if the implementation
    does not support a valid /fontTarget/, FONT_UNAVAILABLE_NV if
    the font is not available (e.g. does not exist on the system), or
    FONT_UNINTELLIGIBLE_NV if the font is available but cannot be loaded
    for some implementation-dependent reason.  FONT_UNAVAILABLE_NV will
    not be returned by PathMemoryGlyphIndexArrayNV because the font
    data is read from system memory.  If the command generates an error,
    that error's enum value will be returned.  For example, an invalid
    value for /fontTarget/ will return INVALID_ENUM.  While the return
    value indicates the error, the error will /also/ be generated in the
    conventional way so GetError will return it and error callbacks are
    generated normally.

    When successful, path names /firstPathName/ through
    /firstPathName+numGlyphs-1/ now are specified as path objects
    corresponding to the sequence of glyphs in the font indicated
    by /fontTarget/, /fontSize/, and /fontData/ for glyph indices
    from /firstGlyphIndex/ to /firstGlyphIndex+numGlyphs-1/ where
    /firstPathName/ corresponds to the glyph index /firstGlyphIndex/
    and onward sequentially.  If a glyph index does not correspond to an
    actual glyph index in the font format, the respective path object is
    left undisturbed.  (It is the application's responsibility to know
    the valid range of glyph indices for the font.)  When unsuccessful
    other than due to an OUT_OF_MEMORY error, no path objects are
    specified or otherwise modified.

    The path objects are created in the same manner described for
    PathGlyphsNV in section 5.X.1.3 (Font Glyph Path Specification)
    except the GLYPH_HAS_KERNING_BIT_NV and FONT_HAS_KERNING_BIT_NV
    metrics are always false (because GetPathSpacingNV applies to
    glyphs specified from Unicode code points).  In particular, the
    /pathParameterTemplate/ and /emScale/ parameters have the same
    interpretation as the PathGlyphsNV command.

    For the PathGlyphIndexArrayNV command, the /fontTarget/ parameter
    must be either SYSTEM_FONT_NAME_NV or FILE_NAME_NV; otherwise the
    INVALID_ENUM error is generated.  The /fontStyle/ parameter is
    a bitfield allowed to have the bits BOLD_BIT_NV or ITALIC_BIT_NV
    set; if other bits are set, the INVALID_VALUE error is generated.
    The interpretation of the /fontTarget/, /fontName/, and /fontStyle/
    parameters is identical to the interpretation described in section
    5.X.1.3 (Font Glyph Path Specification).

    For the PathMemoryGlyphIndexArrayNV command, /fontTarget/ must
    be STANDARD_FONT_FORMAT_NV; otherwise INVALID_ENUM is generated
    (and returned).  STANDARD_FONT_FORMAT_NV implies: /fontSize/ is
    the size of the memory storing the font data in memory; /fontData/
    is a pointer to the beginning of the font data; and /faceIndex/ is
    the index of the face within the font, typically specified as zero.

    The specific standard font formats supported by
    STANDARD_FONT_FORMAT_NV are implementation-dependent, but the TrueType
    format should be supported.  Magic numbers if the font memory data
    are expected to be used to identify the specific font format.

    The INVALID_VALUE error is generated if any of /fontSize/ or
    /faceIndex/ or /emScale/ are negative.

    [NOTE: PathGlyphIndexRangeNV is deprecated in favor of
    PathGlyphIndexArrayNV and PathMemoryGlyphIndexArrayNV.]

    The command

        enum PathGlyphIndexRangeNV(enum fontTarget,
                                   const void *fontName,
                                   bitfield fontStyle,
                                   uint pathParameterTemplate,
                                   float emScale,
                                   uint baseAndCount[2]);

    creates, if successful and no error occurs, a range of path objects
    that correspond to the complete range of glyphs as ordered by glyph
    index in a font face.

    The command returns the value FONT_GLYPHS_AVAILABLE_NV when
    successful; otherwise one of the following values is returned
    depending on the nature of the failure.  The unsuccessful command
    returns the value FONT_TARGET_UNAVAILABLE_NV if the implementation
    does not support a valid /fontTarget/, FONT_UNAVAILABLE_NV if
    the font is not available (e.g. does not exist on the system), or
    FONT_UNINTELLIGIBLE_NV if the font is available but cannot be loaded
    for some implementation-dependent reason.  If the command generates
    an error, that error's enum value will be returned.  For example, an
    invalid value for /fontTarget/ will return INVALID_ENUM.  While the
    return value indicates the error, the error will /also/ be generated
    in the conventional way so GetError will return it and error callbacks
    are generated normally.

    The /fontTarget/ parameter must be either SYSTEM_FONT_NAME_NV
    or FILE_NAME_NV; otherwise the INVALID_ENUM error is generated.
    The interpretation of the /fontTarget/ and /fontName/ parameters
    is identical to the interpretation described in section 5.X.1.3
    (Font Glyph Path Specification).

    The error INVALID_VALUE is generated if /emScale/ is negative.

    The /fontStyle/ parameter is a bitfield allowed to have the
    bits BOLD_BIT_NV or ITALIC_BIT_NV set; if other bits are set, the
    INVALID_VALUE error is generated.

    When successful, elements 0 and 1 of the /baseAndCount/ array
    parameter are written values /B/ and /N/ respectively where the
    path names /B/ through /B+N-1/ are previously unused (i.e. there
    are /N/ previously unused path object names starting at /B/) but
    now are specified as path objects corresponding to the complete set
    of glyphs in the font indicated by /fontTarget/ and /fontName/.
    When unsuccessful (including when any error, even OUT_OF_MEMORY,
    is generated by the command), elements 0 and 1 of the /baseAndCount/
    array parameter are both written to zero.

    The path objects are created in the same manner described for
    PathGlyphsNV in section 5.X.1.3 (Font Glyph Path Specification)
    except the GLYPH_HAS_KERNING_BIT_NV and FONT_HAS_KERNING_BIT_NV
    metrics are always false (because GetPathSpacingNV applies to
    glyphs specified from Unicode code points).  In particular, the
    /pathParameterTemplate/ and /emScale/ parameters have the same
    interpretation as the PathGlyphsNV command.

    5.X.1.4 Path Modification

    Several commands allow the commands and/or coordinates of an existing
    path object to be modified.

    The command

        void PathCoordsNV(uint path,
                          sizei numCoords, enum coordType,
                          const void *coords);

    replaces all the coordinates of an existing path object with a new
    set of coordinates.  /path/ names the path object to modify; the
    error INVALID_OPERATION is generated if /path/ is not an existing
    path object.

    The new path coordinates are read sequentially from the
    /coords/ array.  The type of the coordinates read from the /coords/
    array is determined by the /coordType/ parameter which must be
    one of BYTE, UNSIGNED_BYTE, SHORT, UNSIGNED_SHORT, or FLOAT,
    otherwise the INVALID_ENUM error is generated.

    The INVALID_OPERATION error is generated if /numCoords/ does not
    equal the number of coordinates referenced by the path object's
    existing command sequence (so /numCoords/ provides a sanity check
    that the /coords/ array is being interpreted properly).  The error
    INVALID_VALUE is generated if /numCoords/ is negative.

    If the PathCoordsNV command results in an error, the path object named
    /path/ is not changed; if there is no error, the prior coordinates of
    /path/ are lost.  If there is no error, the commands and parameters
    of the path object are not changed.

    The command

        void PathSubCoordsNV(uint path,
                             sizei coordStart,
                             sizei numCoords, enum coordType,
                             const void *coords);

    replaces a range of the coordinates of an existing path object with
    a new set of coordinates.  /path/ names the path object to modify;
    the error INVALID_OPERATION is generated if /path/ is not an existing
    path object.

    The new path coordinates are read sequentially from the
    /coords/ array.  The type of the coordinates read from the /coords/
    array is determined by the /coordType/ parameter which must be
    one of BYTE, UNSIGNED_BYTE, SHORT, UNSIGNED_SHORT, or FLOAT,
    otherwise the INVALID_ENUM error is generated.

    The coordinates from the /coords/ array replace the coordinates
    starting at coordinate index (zero-based) /coordStart/ through
    /coordStart/+/numCoords/-1 inclusive in the existing path object's
    coordinate array.  If /numCoords/ is zero, no coordinates are changed.
    If /coordStart/+/numCoords/ is greater than the number of coordinates
    in the existing path object, the INVALID_OPERATION error is generated.
    If either /coordStart/ or /numCoords/ is negative, the INVALID_VALUE
    error is generated.

    If the PathCoordsNV command results in an error, the path object named
    /path/ is not changed; if there is no error, the prior coordinates
    within the updated range of /path/ are lost.  If there is no error,
    the commands, coordinates outside the updated range, and parameters
    of the path object are not changed.

    The command

        void PathSubCommandsNV(uint path,
                               sizei commandStart, sizei commandsToDelete,
                               sizei numCommands, const ubyte *commands,
                               sizei numCoords, enum coordType,
                               const void *coords);

    replaces a range of existing commands and their associated coordinates
    with a new sequence of commands and associated coordinates.  /path/
    names the path object to modify; the error INVALID_OPERATION is
    generated if /path/ is not an existing path object.

    The error INVALID_OPERATION is generated if any of /commandStart/,
    /commandsToDelete/, /numCommands/, or /numCoords/ is negative.

    The PathSubCommandsNV command works in two steps.
    First, deleting commands in the range /commandStart/ to
    /commandStart/+/commandsToDelete/-1 inclusive from the existing
    path object.  If /commandsToDelete/ exceeds the number of commands
    from /commandStart/ to the end of the path command sequence,
    all the commands from /commandsToDelete/ on are deleted.  This
    includes deleting the coordinates associated with these commands.
    If /commandsToDelete/ is zero, zero commands and zero coordinates are
    deleted.  Second, /numCommands/ read sequentially from the /commands/
    array are inserted into the existing path object immediately before
    index /commandStart/.  This includes inserting a corresponding number
    of coordinates from the /coords/ array.  If the index /commandStart/
    is greater than the largest valid command index of the path object,
    the commands are simply appended to the end of the path objects
    command and coordinate sequences.

    Each of the /numCommands/ commands in the /command/ array references
    a number of coordinates specified by "Coordinate count" column of
    Table 5.pathCommands, starting with the first (zero) element of
    the /coords/ array and advancing by the coordinate count for each
    command.  If any of these /numCommands/ commands are not listed
    in the "Token" or "Character aliases" columns of Table 5.pathCommands,
    the INVALID_ENUM error is generated.

    The INVALID_OPERATION error is generated if /numCoords/ does not equal
    the number of coordinates referenced by the command sequence to insert
    as specified by /numCommands/ and /commands/ (so /numCoords/ provides
    a sanity check that the /coords/ array is being interpreted properly).
    The error INVALID_VALUE is generated if any of /commandStart/,
    /commandsToDelete/, /numCommands/ or /numCoords/ are negative.

    The type of the coordinates in the /coords/ array is specified
    by /coordType/ and must be one of BYTE, UNSIGNED_BYTE, SHORT,
    UNSIGNED_SHORT, or FLOAT; otherwise the INVALID_ENUM error is
    generated.

    If the PathSubCommandsNV command results in an error, the path
    object named /path/ is not changed; if there is no error, the prior
    (now deleted) commands and coordinates within the updated range of
    /path/ are lost.  If there is no error, the commands, coordinates
    outside the deleted range, and parameters of the path object are not
    changed though commands and coordinates indexed beyond /commandStart/
    are shifted in their sequence within the path object to make room
    in the command and coordinate arrays for the newly inserted commands
    and coordinates.

    5.X.1.5 Path Parameter Specification

    Each path object has its own set of path parameters that control
    how the path object is filled and stroked when stenciled and covered.

    Table 5.pathParameters

        Name                             Type     Required Values or Range
        -------------------------------  -------  -----------------------------------------------
        PATH_STROKE_WIDTH_NV             float    non-negative
        PATH_INITIAL_END_CAP_NV          enum     FLAT, SQUARE_NV, ROUND_NV, TRIANGULAR_NV
        PATH_TERMINAL_END_CAP_NV         enum     FLAT, SQUARE_NV, ROUND_NV, TRIANGULAR_NV
        PATH_INITIAL_DASH_CAP_NV         enum     FLAT, SQUARE_NV, ROUND_NV, TRIANGULAR_NV
        PATH_TERMINAL_DASH_CAP_NV        enum     FLAT, SQUARE_NV, ROUND_NV, TRIANGULAR_NV
        PATH_JOIN_STYLE_NV               enum     MITER_REVERT_NV, MITER_TRUNCATE_NV, BEVEL_NV, ROUND_NV, NONE
        PATH_MITER_LIMIT_NV              float    non-negative
        PATH_DASH_OFFSET_NV              float    any value
        PATH_DASH_OFFSET_RESET_NV        enum     MOVE_TO_RESET_NV, MOVE_TO_CONTINUES_NV
        PATH_CLIENT_LENGTH_NV            float    non-negative
        PATH_FILL_MODE_NV                enum     COUNT_UP_NV, COUNT_DOWN_NV, INVERT
        PATH_FILL_MASK_NV                integer  any value
        PATH_FILL_COVER_MODE_NV          enum     CONVEX_HULL_NV, BOUNDING_BOX_NV
        PATH_STROKE_COVER_MODE_NV        enum     CONVEX_HULL_NV, BOUNDING_BOX_NV
        PATH_STROKE_MASK_NV              integer  any value
        PATH_STROKE_BOUND_NV             float    any value in [0.0,1.0]

    The commands

        void PathParameterivNV(uint path, enum pname, const int *value);
        void PathParameteriNV(uint path, enum pname, int value);
        void PathParameterfvNV(uint path, enum pname, const float *value);
        void PathParameterfNV(uint path, enum pname, float value);

    specify the value of path parameters for the specified path object
    named /path/.  The error INVALID_OPERATION is generated if /path/
    is not an existing path object.

    Each parameter has a single (scalar) value.

    /pname/ must be one of the tokens in the "Name" column of
    Table 5.pathParameters, PATH_END_CAPS_NV, or PATH_DASH_CAPS_NV.
    The required values or range of each allowed parameter name token
    is listed in Table 5.pathParameter's "Required Values/Range" column.

    For values of /pname/ listed in Table 5.pathsParameters, the specified
    parameter is specified by /value/ when /value/ is a float or int,
    or if /value/ is a pointer to a float or int, accessed through that
    pointer.  The error INVALID_VALUE is generated if the specified
    value is negative for parameters required to be non-negative in
    Table 5.pathParameters.  Values specified to be clamped to the [0,1] range
    in Table 5.pathParameters are so clamped prior to setting the
    specified path parameter to that clamped value.

    The /pname/ of PATH_END_CAPS_NV is handled specially and updates
    /both/ the PATH_INITIAL_END_CAP_NV and PATH_TERMINAL_END_CAP_NV
    parameters of the path with the specified value.  The /pname/
    of PATH_DASH_CAPS_NV is handled specially and updates /both/ the
    PATH_INITIAL_DASH_CAP_NV and PATH_TERMINAL_DASH_CAP_NV parameters
    of the path with the specified value.

    The error INVALID_VALUE is generated if the specified parameter value
    is not within the require range for parameters typed float or integer.
    The error INVALID_ENUM is generated if the specified parameter value
    is not one of the listed tokens for parameters typed enum.

    The dash pattern of a path object consists of a sequence of path-space
    lengths of alternating "on" and "off" dash segments.  The first
    value of the dash array defines the length, in path space, of the
    first "on" dash segment.  The second value defines the length of the
    following "off" segment.  Each subsequent pair of values defines one
    "on" and one "off" segment.

    Parameters to control the dash pattern of a stroked path are specified
    by the command

        void PathDashArrayNV(uint path,
                             sizei dashCount, const float *dashArray);

    where /path/ is the name of an existing path object.  The error
    INVALID_OPERATION is generated if /path/ is not an existing path
    object.

    A /dashCount/ of zero indicates the path object is not dashed; in
    this case, the /dashArray/ is not accessed.  Otherwise, /dashCount/
    provides a count of how many float values to read from the /dashArray/
    array.  If any of the /dashCount/ elements of /dashArray/ are
    negative, the INVALID_VALUE error is generated.

    If /dashCount/ is negative, the INVALID_VALUE error is generated.

    If an error occurs, the path object's existing dash pattern state
    is not changed.

    The path parameters of a newly specified path object are initialized
    as specified in Table 6.Y.

    5.X.1.6 Path Weighting, Interpolation, and Copying

    The command

        void WeightPathsNV(uint resultPath,
                           sizei numPaths,
                           const uint paths[], const float weights[]);

    linearly combines, as appropriate, the /numPaths/ path objects in
    the array paths based on each path object's respective weight from
    the weights array.  The resulting path creates or replaces the
    path object /resultPath/.  The INVALID_VALUE error is generated if
    /numPaths/ is less than one.

    If the /resultPath/ name also names one of the paths in the /paths/
    array, the path resulting from the linear combination of paths
    replaces the source path also named /resultPath/ but not until after
    the linear combination path has been determined.

    This command requires all the paths in the paths array to
    be /consistent/; otherwise the INVALID_OPERATION error is
    generated.  For all the paths to be /consistent/, all /numPaths/ paths
    in the /paths/ array must have the identical count of commands and
    each corresponding /i/th command in each path must have the identical
    command type.

    However the arc commands (specifically SMALL_CCW_ARC_TO_NV,
    RELATIVE_SMALL_CCW_ARC_TO_NV, SMALL_CW_ARC_TO_NV,
    RELATIVE_SMALL_CW_ARC_TO_NV, LARGE_CCW_ARC_TO_NV,
    RELATIVE_LARGE_CCW_ARC_TO_NV, LARGE_CW_ARC_TO_NV,
    RELATIVE_LARGE_CW_ARC_TO_NV, CIRCULAR_CCW_ARC_TO_NV,
    CIRCULAR_CW_ARC_TO_NV, CIRCULAR_TANGENT_ARC_TO_NV, ARC_TO_NV, and
    RELATIVE_ARC_TO_NV) can not be weighted because the linear combination
    of the curves these arc commands generate do not generally result in
    a command of the same form; so if any of these arc commands appears
    in a path object passed to WeightPathsNV the INVALID_OPERATION error
    is generated.

    The weighted path has a command sequence identical to any of the
    input path objects to be weighted (since all the input path command
    sequences are required to be identical).

    The weighted path has a coordinate sequence constructed by weighting
    each correspondingly indexed coordinate /i/ for all paths indexed by
    /j/ from zero to /numPaths/-1 in the /paths/ array.  Each coordinate
    /i/ from path /j/ is weighted by the weight in /weights/ indexed
    by /j/.

    The path parameters for the weighted path are copied from the path
    named by the first (0th) element of the /paths/ array.  The path
    metric values (as queried by GetPathMetricsNV in section 6.X.3)
    are all specified to be -1 for the newly specified path object
    (ignoring the path metrics for all the input path objects).
    Kerning information (as queriable by GetPathSpacingNV in section
    6.X.3) is also not copied.

    The command

        void InterpolatePathsNV(uint resultPath,
                                uint pathA, uint pathB,
                                float weight);

    is equivalent to

        uint paths[2] = { pathA, pathB };
        float weights[2] = { 1-weight, weight };
        WeightPathsNV(resultPath, 2, paths, weights);

    The command

        void CopyPathNV(uint resultPath, uint srcPath);

    copies the path object named /srcPath/ to the path object named
    /resultPath/.  The error INVALID_OPERATION is generated if /srcPath/
    does not exist.  The outline (commands and coordinates), parameters,
    and glyph metrics and kerning information (if they exist) are all
    copied without change.

    5.X.1.7 Path Transformation

    The command

        void TransformPathNV(uint resultPath,
                             uint srcPath,
                             enum transformType,
                             const float *transformValues);

    transforms the path object named /srcPath/ by the transform specified
    by the /transformType/ and its associated /transformValues/.
    The resulting path creates or replaces the path object /resultPath/.

    If the /resultPath/ and /srcPath/ names are identical, the path resulting
    from the transform replaces the name after the source path is transformed.

    The /transformType/ must be one of NONE, TRANSLATE_X_NV,
    TRANSLATE_Y_NV, TRANSLATE_2D_NV, TRANSLATE_3D_NV, AFFINE_2D_NV,
    AFFINE_3D_NV, TRANSPOSE_AFFINE_2D_NV, or TRANSPOSE_AFFINE_3D_NV.

        transformType               Matrix
        --------------------------  -------------------------
        NONE                        [   1   0   0   0 ]
                                    [   0   1   0   0 ]
                                    [   0   0   1   0 ]
                                    [   0   0   0   1 ]

        TRANSLATE_X_NV              [   1   0   0  v0 ]
                                    [   0   1   0   0 ]
                                    [   0   0   1   0 ]
                                    [   0   0   0   1 ]

        TRANSLATE_Y_NV              [   1   0   0   0 ]
                                    [   0   1   0  v0 ]
                                    [   0   0   1   0 ]
                                    [   0   0   0   1 ]

        TRANSLATE_2D_NV             [   1   0   0  v0 ]
                                    [   0   1   0  v1 ]
                                    [   0   0   1   0 ]
                                    [   0   0   0   1 ]

        TRANSLATE_3D_NV             [   1   0   0  v0 ]
                                    [   0   1   0  v1 ]
                                    [   0   0   1  v2 ]
                                    [   0   0   0   1 ]

        AFFINE_2D_NV                [  v0  v2   0  v4 ]
                                    [  v1  v3   0  v5 ]
                                    [   0   0   1   0 ]
                                    [   0   0   0   1 ]

        TRANSPOSE_AFFINE_2D_NV      [  v0  v1   0  v2 ]
                                    [  v3  v4   0  v5 ]
                                    [   0   0   1   0 ]
                                    [   0   0   0   1 ]

        AFFINE_3D_NV                [  v0  v3  v6  v9 ]
                                    [  v1  v4  v7 v10 ]
                                    [  v2  v5  v8 v11 ]
                                    [   0   0   0   1 ]

        TRANSPOSE_AFFINE_3D_NV      [  v0  v1  v2  v3 ]
                                    [  v4  v5  v6  v7 ]
                                    [  v8  v9 v10 v11 ]
                                    [   0   0   0   1 ]

    Table 5.transformType:  Mapping from /transformType/ to a 4x4
    transform matrix where v/i/ is the ith (base 0) element of the
    /transformValues/ array.

    The transformation of a path proceeds path command by path command.
    Each path command results in a transformed path command equivalent
    to what would happen if every point on the path command segment were
    transformed by the transform from Table 5.transformType and had a
    projective normalization applied.

    Commands with absolute control points have their control points
    transformed by the effective 4x4 projective matrix, and the resulting
    x & y coordinates serve as the transformed command's respective
    control point.

    Control points of relative commands are first made into absolute
    coordinates given the command's current control point, transformed
    in the same manner as an absolute control point, and then adjusted
    back to relative to their transformed current control point.

    Horizontal and vertical line to commands are promoted to corresponding
    "line to" commands if the transformed command is not an exactly
    horizontal or vertical command respectively after transformation;
    otherwise, these commands are not promoted but may transition from
    horizontal to vertical or vice versa as the case may be.

    Commands for partial elliptical arcs generate an equivalent new
    transformed arc.

    XXX more detail/math about arcs?

    The CIRCULAR_CCW_ARC_TO_NV and CIRCULAR_CW_ARC_TO_NV commands are
    converted to transformed *_ARC_TO_NV commands if the transformed
    circular arc is itself not a circular arc.

    The CIRCULAR_TANGENT_ARC_TO_NV command is converted into a LINE_TO_NV
    command and *_ARC_TO_NV command if the transformed circular arc is
    itself not a circular arc.

    The CLOSE_PATH_NV and RESTART_PATH_NV (having no control points)
    are undisturbed by path transformation.  The order of path commands
    is invariant under path transformation.

    5.X.1.8  Path Name Management

    The command

        uint GenPathsNV(sizei range);

    returns an integer /n/ such that names /n/, ..., /n+range-1/ are
    previously unused (i.e. there are /range/ previously unused path object
    names starting at /n/).  These names are marked as used, for the
    purposes of subsequent GenPathsNV only, but they do not acquire
    path object state until each particular name is used to specify
    a path object.

    Path objects are deleted by calling

        void DeletePathsNV(uint path, sizei range);

    where /path/ contains /range/ names of path objects to be delete.
    After a path object is deleted, its name is again unused.  Unused
    names in /paths/ are silently ignored.

    The query

        boolean IsPathNV(uint path);

    returns TRUE if /path/ is the name of a path object.  If path is
    not the name of a path object, or if an error condition occurs,
    IsPathNV returns FALSE.  A name retuned by GenPathsNV, but without
    a path specified for it yet, is not the name of a path object.

    5.X.2 Path Rendering

    Path objects update the framebuffer through one of two processes:
    "stenciling" that updates /just/ the stencil buffer with the path's
    coverage information, and "covering" that rasterizes fragments into
    the framebuffer for a region guaranteed to cover the region of path
    coverage updated by stenciling, assuming the same path object,
    fill mode or stroking parameters, transformation state, and set of
    accessible samples (as will be explained).

    5.X.2.1 Path Stenciling

    STENCILING FILLED PATHS

    The command

        void PathStencilFuncNV(enum func, int ref, uint mask);

    configures the stencil function, stencil reference value, and stencil
    read mask to be used by the StencilFillPathNV and StencilStrokePathNV
    commands described subsequently.  The parameters accept the same
    values allowed by the StencilFunc command.

    The command

        void PathStencilDepthOffsetNV(float factor, float units);

    configures the depth offset factor and units state (see section 3.6.4)
    to be used by the StencilFillPathNV and StencilStrokePathNV commands
    described subsequently.

    The command

        void StencilFillPathNV(uint path,
                               enum fillMode, uint mask);

    transforms into window space the outline of the path object named
    /path/ based on the current modelview, projection, viewport,
    and depth range transforms (ignoring any vertex and/or geometry
    shader or program that might be active/enabled) and then updates
    the stencil values of all /accessible samples/ (explained below) in
    the framebuffer.  Each sample's stencil buffer value is updated based
    on the winding number of that sample with respect to the transformed
    outline of the path object with any non-closed subpath forced closed
    and the specified /fillMode/.

    If /path/ does not name an existing path object, the command does
    nothing (and no error is generated).

    If the path's command sequence specifies unclosed subpaths (so not
    contours) due to MOVE_TO_NV commands, such subpaths are trivially
    closed by connecting with a line segment the initial and terminal
    control points of each such path command subsequence.

    Transformation of a path's outline works by taking all positions
    on the path's outline in 2D path space (x,y) and constructing an
    object space position (x,y,0,1) that is then used as the (xo,yo,zo,wo)
    position in section 2.12 ("Fixed-Function Vertex Transformation")
    to compute corresponding eye-space coordinates (xe,ye,ze,we) and
    clip-space coordinates (xc,yc,zc,wc).  A path outline's clip-space
    coordinates are further transformed into window space as described in
    section 2.16 ("Coordinate Transformations").  This process provides a
    mapping 2D path coordinates to 2D window coordinates and depth values.
    The resulting 2D window coordinates are undefined if any of the
    transformations involved are singular or may be inaccurate if any
    of the transformations (or their combination) are ill-conditioned.

    The winding number for a sample with respect to the path outline,
    transformed into window space, is computed by counting the (signed)
    number of revolutions around the sample point when traversing each
    (trivially closed if necessary) contour once in the transformed path.
    This traversal is performed in the order of the path's command
    sequence.  Starting from an initially zero winding count, each
    counterclockwise revolution when the front face mode is CCW (or
    clockwise revolution when the front face mode is CW) around the sample
    point increments the winding count by one; while each clockwise
    revolution when the front face mode is CCW (or counterclockwise
    revolution when the front face mode is CW) around the sample point
    decrements the winding count by one.

    The /mask/ parameter controls what subset of stencil bits are affected
    by the command.  If the /mask/ parameter is zero, the path object's
    fill mask parameter (PATH_FILL_MASK_NV) is considered the effective
    value of /mask/.

    The /fillMode/ parameter must be one of INVERT, COUNT_UP_NV,
    COUNT_DOWN_NV, or PATH_FILL_MODE_NV; otherwise the INVALID_ENUM error
    is generated.  INVERT inverts the bits set in the effective /mask/
    value for each sample's stencil value if the winding number for the
    given sample is odd.  COUNT_UP_NV adds with modulo n arithmetic the
    winding number of each sample with the sample's prior stencil buffer
    value; the result of this addition is written into the sample's
    stencil value but the bits of the stencil value not set in the
    effective /mask/ value are left unchanged.  COUNT_DOWN_NV subtracts
    with modulo /n/ arithmetic the winding number of each sample with the
    sample's prior stencil buffer value; the result of this subtraction is
    written into the sample's stencil value but the bits of the stencil
    value not set in the effective /mask/ value are left unchanged.
    PATH_FILL_MODE_NV uses the path object's counting mode parameter
    (one of INVERT, COUNT_UP_NV, or COUNT_DOWN_NV).

    The value of /n/ for the modulo /n/ arithmetic used by COUNT_UP_NV
    and COUNT_DOWN_NV is the effective /mask/+1.  The error INVALID_VALUE
    is generated if the specified /fillMode/ is COUNT_UP_NV or
    COUNT_DOWN_NV and the specified /mask/+1 is not an integer power
    of two.  If the /fillMode/ is PATH_FILL_MODE_NV; the path object's
    counting mode parameter is COUNT_UP_NV or COUNT_DOWN_NV; and the
    effective mask+1 value is not an integer power of two, treat the
    mask as zero (effectively meaning no stencil bits will be modified).

    ACCESSIBLE SAMPLES WITH RESPECT TO A TRANSFORMED PATH

    The accessible samples of a transformed path that are updated are
    the samples that remain after discarding the following samples:

        *   Any sample that would be clipped as specified in section 2.22
            ("Primitive Clipping") because its corresponding position in
            clip space (xc,yc,zc,wc) or (xe,ye,ze,we) would be clipped
            by the clip volume or enabled client-defined clip planes.

        *   Any sample that would not be updated during polygon rendering
            due to polygon stipple (section 3.6.2) if POLYGON_STIPPLE
            is enabled.

        *   Any sample that would fail the pixel ownership test (section
            4.1.1) if rasterized.

        *   Any sample that would fail the scissor test (section 4.1.2)
            if SCISSOR_TEST is enabled.

        *   Any sample that would fail the depth test (section 4.1.6)
            if DEPTH_TEST is enabled where the fragment depth for the
            depth test comes from the depth plane of the path when
            transformed by the modelview, projection, viewport, and
            depth range transforms and depth offset (section 3.6.4)
            has been applied based on the slope of this plane operating
            as if POLYGON_OFFSET_FILL is forced enabled and using the
            factor and units parameters set by PathStencilDepthOffsetNV
            (rather than the state set by PolygonOffset).

        *   Any sample that would fail the depth bounds test (section
            4.1.X in EXT_depth_bounds_test specification) if
            DEPTH_BOUNDS_TEST_EXT is enabled.

    And for the StencilFillPathNV and StencilStrokePathNV commands (so
    not applicable to the CoverFillPathNV and CoverStrokePathNV commands):

        *   Any sample that would fail the (implicitly enabled) stencil
            test (section 4.1.5) with the stencil function configured
            based on the path stencil function state configured by
            PathStencilFuncNV.  In the case of the StencilFillPathNV
            and StencilStrokePathNV commands and their instanced
            versions (section 5.X.2.3), the effective stencil read
            mask for the stencil mask is treated as the value of
            PATH_STENCIL_VALUE_MASK bit-wise ANDed with the bit-invert
            of the effective /mask/ parameter value; otherwise, for the
            cover commands, the stencil test operates normally.  In the
            case the stencil test fails during a path stencil operation,
            the stencil fail operation is ignored and the pixel's stencil
            value is left undisturbed (as if the stencil operation was
            KEEP).

        *   The state of the face culling (CULL_FACE) enable is ignored.

    STENCILING STROKED PATHS

    The command

        void StencilStrokePathNV(uint path,
                                 int reference, uint mask);

    transforms into window space the stroked region of the path object
    named /path/ based on the current modelview, projection, viewport,
    and depth range transforms (ignoring any vertex and/or geometry
    shader or program that might be active/enabled) and then updates
    the stencil values of a subset of the accessible samples (see above)
    in the framebuffer.

    If /path/ does not name an existing path object, the command does
    nothing (and no error is generated).

    The path object's stroke width parameter (PATH_STROKE_WIDTH_NV) in
    path space units determines the width of the path's stroked region.

    When the dash array count of a path object is zero (dashing is
    considered subsequently), the stroke of a transformed path's outline
    is the region of window space defined by the union of:

        *   Sweeping an orthogonal centered line segment of the (above
            determined) effective stroke width along each path segment
            in the path's transformed outline.

        *   End cap regions (explained below) appended to the initial
            and terminal control points of non-closed command sequences
            in the path.  For a sequence of commands that form a closed
            contour, the end cap regions are ignored.

        *   Join style regions (explained below) between connected path
            segments meet.

    Any accessible samples within the union of these three regions are
    considered within the path object's stroke.

    The /mask/ parameter controls what subset of stencil bits are affected
    by the command.  If the /mask/ parameter is zero, the path object's
    stroke mask parameter (PATH_STROKE_MASK_NV) is considered the effective
    value of /mask/.

    A sample's stencil bits that are set in the effective /mask/ value
    are updated with the specified stencil /reference/ value if the
    sample is accessible (as specified above) and within the stroke of
    the transformed path's outline.

    Every path object has an initial and terminal end cap parameter
    (PATH_INITIAL_END_CAP_NV and PATH_TERMINAL_END_CAP_NV) that is
    one of FLAT, SQUARE_NV, ROUND_NV, or TRIANGULAR_NV.  There are no
    samples within a FLAT end cap.  The SQUARE_NV cap extends centered
    and tangent to the given end (initial or terminal) of the subpath
    for half the effective stroke width; in other words, a square cap
    is a half-square that kisses watertightly the end of a subpath.
    The ROUND_NV cap appends a semi-circle, centered and tangent,
    with the diameter of the effective stroke width to the given end
    (initial or terminal) of the subpath; in other words, a round cap
    is a semi-circle that kisses watertightly the end of a subpath.
    The TRIANGULAR_NV cap appends a right triangle, centered and tangent,
    with its hypotenuse flush to the given end of the subpath; in other
    words, a triangular cap is a right triangle that kisses watertightly
    the end of a subpath with the triangle's longest side.

    Every path object has a join style parameter (PATH_JOIN_STYLE_NV)
    that is one of BEVEL_NV, ROUND_NV, MITER_REVERT_NV, MITER_TRUNCATE_NV,
    or NONE; each path object also has a miter limit value.  The BEVEL_NV
    join style inserts a triangle with two vertices at the outside
    corners where two connected path segments join and a third vertex at
    the common end point shared by the two path segments.  The ROUND_NV
    join style inserts a wedge-shaped portion of a circle centered at
    the common end point shared by the two path segments; the radius of
    the circle is half the effective stroke width.  There are no samples
    within a NONE join style.  The MITER_REVERT_NV join style inserts a
    quadrilateral with two opposite vertices at the outside corners where
    the two connected path segments join and two opposite vertices with
    one on the path's junction between the two joining path segments and
    the other at the common end point shared by the two path segments.
    However, the MITER_REVERT_NV join style behaves as the BEVEL_NV
    style if the sine of half the angle between the two joined segments
    is less than the path object's PATH_STROKE_WIDTH value divided by
    the path's PATH_MITER_LIMIT_NV value.  The MITER_TRUNCATE_NV join
    style is similar to MITER_REVERT_NV but rather than reverting to a
    bevel when the miter limit is exceeded, instead the tip of the miter
    quadrilateral is truncated such that the miter does not extend beyond
    the miter limit.

    When the dash array count of a path object is /not/ zero, the path is
    broken up into a sequence of paths based on the path object's dash
    array count, dash array, dash offset, and dash cap parameters (see
    section 5.X.1.5).  This sequence of paths are handled as if their
    dash count array is zero so their stroked region can be determined
    for this stroking case that has already been explained.

    The dash pattern defined by the dash array is a sequence of lengths of
    alternating "on" and "off" dash segments.  The first (0th) element of
    the dash array defines the length, in path space, of the first "on"
    dash segment.  The second value defines the length of the following
    "off" segment.  Each subsequent pair of values defines on "on"
    and one "off" segment.

    The initial cap of the first dash segment uses the path's initial
    dash cap style state (PATH_INITIAL_END_CAP_NV) as the effective
    initial end cap for this first dash segment; the terminal cap
    of the last dash segment uses the path's terminal dash cap style
    state (PATH_TERMINAL_END_CAP_NV) as the effective terminal cap for
    this last dash segment; all other caps of dash segments use the
    PATH_INITIAL_DASH_CAP_NV for the initial cap of the segment and the
    PATH_TERMINAL_DASH_CAP_NV for the terminal cap of the segment.

    The MOVE_TO_RESETS_NV value for a path's dash offset reset parameter
    (PATH_DASH_OFFSET_RESET_NV) means that the dash offset resets to the
    path's dash offset parameter upon a MOVE_TO_NV, RELATIVE_MOVE_TO_NV,
    RESTART_PATH_NV, or RECT_NV command (an command that does an implicit
    or explicit move-to) while dashing the path's command sequence.
    The MOVE_TO_CONTINUES_NV value means that the dash pattern
    progresses normally (without reset) when dashing a MOVE_TO_NV or
    RELATIVE_MOVE_TO_NV command.

    Every path object has a stroke approximation bound parameter
    (PATH_STROKE_BOUND_NV) that is a floating-point value /sab/ clamped
    between 0.0 and 1.0 and set and queried with the PATH_STROKE_BOUND_NV
    path parameter.  Exact determination of samples swept an orthogonal
    centered line segment along cubic Bezier segments and rational
    quadratic Bezier curves (so non-circular partial elliptical arcs) is
    intractable for real-time rendering so an approximation is required;
    /sab/ intuitively bounds the approximation error as a percentage of
    the path object's stroke width.  Specifically, this path parameter
    requests the implementation to stencil any samples within /sweep/
    object space units of the exact sweep of the path's cubic Bezier
    segments or partial elliptical arcs to be sampled by the stroke where

      sweep = ((1-sab)*sw)/2

    where /sw/ is the path object's stroke width.  The initial value
    of /sab/ when a path is created is 0.2.  In practical terms, this
    initial value means the stencil sample positions coverage within 80%
    (100%-20%) of the stroke width of cubic and rational quadratic stroke
    segments should be sampled.

    If the path object's client length parameter (PATH_CLIENT_LENGTH_NV)
    value /clen/ is non-zero, prior to generating the dashed segments, the
    dash pattern and dash offset lengths should be scaled by (multiplied
    by) the clen/plen where /plen/ is the path object's computed length
    (PATH_COMPUTED_LENGTH_NV).

    5.X.2.2 Path Covering

    COVERING FILLED PATHS

    The command

        void PathCoverDepthFuncNV(enum zfunc);

    configures the depth function to be used by the CoverFillPathNV and
    CoverStrokePathNV commands described subsequently.  The /zfunc/ parameter
    accepts the same values allowed by the DepthFunc command.

    The command

        void CoverFillPathNV(uint path, enum coverMode);

    transforms into window space the outline of the path object named
    /path/ based on the current modelview, projection, viewport,
    and depth range transforms (ignoring any vertex and/or geometry
    shader or program that might be active/enabled) and rasterizes a
    subset of the accessible samples in the framebuffer guaranteed to
    include all samples that would be have a net stencil value change if
    StencilFillPathNV were issued with the same modelview, projection,
    and viewport state.  During this rasterization, the stencil test
    operates normally and as configured; the expectation is the stencil
    test will be used to discard samples not determined "covered" by a
    prior StencilFillPathNV command.  The depth function, if DEPTH_TEST is
    enabled, during this rasterization uses the function specified by
    PathCoverDepthFuncNV (instead of the state specified by DepthFunc).

    If /path/ does not name an existing path object, the command does
    nothing (and no error is generated).

    /coverMode/ must be one of CONVEX_HULL_NV, BOUNDING_BOX_NV, or
    PATH_FILL_COVER_MODE_NV.  The PATH_FILL_COVER_MODE_NV uses the path
    object's PATH_FILL_COVER_MODE_NV parameter value as the effective
    fill cover mode of the cover command.

    When /coverMode/ is CONVEX_HULL_NV or BOUNDING_BOX_NV, the subset
    of accessible pixels that are rasterized are within a convex
    hull or bounding box respectively (each expected to be reasonably
    tight) surrounding all the samples guaranteed to be rasterized by
    CoverFillPathNV.  The bounding box must be orthogonally aligned
    to the path space coordinate system.  (The area of the bounding
    box in path space is guaranteed to be greater than or equal the
    area of the convex hull in path space.) Each rasterized sample
    will be rasterized once and exactly once when CONVEX_HULL_NV or
    BOUNDING_BOX_NV is specified.

    While samples with a net stencil change /must/ be rasterized,
    implementations are explicitly allowed to vary in the rasterization
    of samples for which StencilFillPathNV would /not/ change sample's
    net stencil value.  This means implementations are allowed to (and,
    in fact, are expected to) conservatively "exceed" the region strictly
    stenciled by the path object.

    CoverFillPathNV /requires/ the following rasterization invariance:
    calling CoverFillPathNV for the same (unchanged) path object with
    fixed (unchanged) modelview, projection, and viewport transform state
    with the same (unchanged) set of accessible samples will rasterize
    the exact same set of samples with identical interpolated values
    for respective fragment/sample locations.

    COVERING STROKED PATHS

    The command

        void CoverStrokePathNV(uint path, enum coverMode);

    operates in the same manner as CoverFillPathNV except the region
    guaranteed to be rasterized is, rather than the region within /path/'s
    filled outline, instead the region within the /path/'s stroked region
    as determined by StencilStrokePathNV.  During this rasterization,
    the stencil test operates normally and as configured; the expectation
    is the stencil test will be used to discard samples not determined
    "covered" by a prior StencilStrokePathNV command.  As with CoverFillPathNV,
    the depth function, if DEPTH_TEST is enabled, uses the function specified
    by PathCoverDepthFuncNV.

    /coverMode/ must be one of CONVEX_HULL_NV, BOUNDING_BOX_NV, or
    PATH_STROKE_COVER_MODE_NV.  The PATH_STROKE_COVER_MODE_NV uses
    the path object's PATH_STROKE_COVER_MODE_NV parameter value as the
    effective stroke cover mode of the cover command.

    If /path/ does not name an existing path object, the command does
    nothing (and no error is generated).

    Analogous to the rasterization guarantee of CoverFillPathNV with
    respect to StencilFillPathNV, CoverStrokePathNV guarantees that all
    samples rasterized by StencilStrokePathNV, given the same transforms
    and accessible pixels and stroke width, will also be rasterized by
    the corresponding CoverStrokePathNV.

    CoverStrokePathNV /requires/ the following rasterization invariance:
    calling CoverStrokePathNV for the same (unchanged) path object with
    fixed (unchanged) modelview, projection, and viewport transform
    state and with the same (unchanged) set of accessible samples will
    rasterize the exact same set of samples with identical interpolated
    values for respective fragment/sample locations.

    PATH COVERING RASTERIZATION DETAILS

    The GL processes fragments rasterized by path cover commands in
    much the same manner as fragments generated by conventional polygon
    rasterization.  However path rendering /ignores/ the following
    operations:

        *  Interpolation of per-vertex data (section 3.6.1).  Path
           primitives have neither conventional vertices nor per-vertex
           data.  Instead fragments generate interpolated per-fragment
           colors, texture coordinate sets, and fog coordinates as a
           linear function of object-space or eye-space path coordinate's
           or using the current color, texture coordinate set, or fog
           coordinate state directly.

        *  Polygon smooth (section 3.6.3).

        *  Polygon mode (section 3.6.4).  Fragments generated by path
           covering never result from point or line rasterizations.

    Polygon stippling (section 3.6.2), depth offset (section 3.6.5), and
    polygon multisample rasterization (3.6.6) do apply to path covering.

    Front and back face determination (explained in section 3.6.1 for
    polygons) operates somewhat differently for transformed paths than
    polygons.  The path's convex hull, bounding box, or multiple hulls
    (depending on the /coverMode/) are specified to wind counterclockwise
    in object space though the transformation of the convex hull into
    window space could reverse this winding.  Whether the GL's front face
    state is CCW or CCW (as set by the FrontFace command) determines
    if the path is front facing or not.  Because the specific vertices
    that belong to the covering geometry are implementation-dependent,
    when the signed area of the covering geometry (computed with equation
    3.8) is sufficiently near zero, the facingness of the path in such
    situations is ill-defined.

    The determination of whether a path transformed into window space is
    front facing or not affects face culling if enabled (section 3.6.1),
    the gl_FrontFacing built-in variable (section 3.12.22), and separate
    (two-sided) stencil testing (section 4.1.5).

    Once fragments have been generated by path covering, the fragments
    are shaded in the same manner as fragments generated by polygon
    rasterization with the following exception:  If a GLSL program is
    in use, any vertex or geometry shader linked into the GLSL program
    is ignored. The fragment shader operates normally except that
    user-defined inputs to the fragment shader behave as specified by
    ProgramPathFragmentInputGenNV. When supported, the fragment shader
    could also use the built-in varying inputs: gl_Texcoord[i],
    gl_Color, gl_SecondaryColor, and gl_FogFragCoord.

    COLOR GENERATION FOR PATH COVER COMMANDS

    The command

        void PathColorGenNV(enum color,
                            enum genMode,
                            enum colorFormat, const float *coeffs);

    controls how the primary and secondary interpolated colors are
    computed for fragment shading operations that occur as a result of
    CoverFillPathNV or CoverStrokePathNV.

    /color/ must be one of PRIMARY_COLOR,
    PRIMARY_COLOR_NV, SECONDARY_COLOR_NV to specify the indicated color
    generation state for the primary, primary, and secondary color
    respectively; otherwise INVALID_ENUM is generated.

    /genMode/ must be one of NONE, OBJECT_LINEAR,
    PATH_OBJECT_BOUNDING_BOX_NV, EYE_LINEAR, or CONSTANT; otherwise
    INVALID_ENUM is generated.

    NONE means the color is not generated but rather uses the
    corresponding color's current color state.  OBJECT_LINEAR means that
    the specified color is generated from a linear combination of the 2D
    path coordinates (x,y).  EYE_LINEAR means the specified color
    is generated from a linear combination of path's 2D coordinates
    transformed in eye space, so (xe, ye, ze, we) from section 2.12
    ("Fixed-Function Vertex Transformation").  CONSTANT means the specified
    color is generated from the specified constant coefficients.

    When covering single paths with CoverFillPathNV or CoverStrokePathNV,
    PATH_OBJECT_BOUNDING_BOX_NV means the specified color is generated
    as a function of object-space (x,y) coordinate normalized to the
    [0..1]x[0..1] range where (0,0) is the corner of the path object's
    bounding box with the minimum x and minimum y coordinates and (1,1)
    is the corner of the path object's bounding box with the maximum x and
    maximum y coordinates.  Using PATH_OBJECT_BOUND_BOX_NV for /genMode/
    means generated colors are undefined if either the width or height
    of the path's bounding box is zero.

    When covering instanced paths with CoverFillPathInstancedNV or
    CoverStrokePathInstancedNV using the BOUNDING_BOX_OF_BOUNDING_BOXES_NV
    cover mode (see section 5.X.2.3), the specified color is generated
    as a function of object-space (x,y) coordinate normalized to the
    [0..1]x[0..1] range where (0,0) is the corner of the bounding box of
    the union of bounding boxes of the set of instanced path objects and
    (1,1) is the corner of the same union bounding box with the maximum
    x and maximum y coordinates.

    When /genMode/ is NONE, then /colorFormat/ must be NONE;
    otherwise INVALID_ENUM is generated.  When /genMode/ is not NONE,
    then /colorFormat/ must be one of LUMINANCE, ALPHA, INTENSITY,
    LUMINANCE_ALPHA, RGB, or RGBA; otherwise INVALID_ENUM is generated.

    In the following equations used for path color generation, coeffs[i]
    is the /i/th element (base zero) of the /coeffs/ array; Rc, Gc,
    Bc, and Aa are the red, green, blue, and alpha colors of the current
    primary or secondary color (depending on the color parameter) when the
    path is covered; and x, y, z, and w are determined by the /genMode/.

    When /genMode/ is EYE_LINEAR, xcoeffs[i] is the /i/th element (base
    zero) of a /xcoeffs/ array generated by multiplying each respective
    vector of four elements of coeffs by the current inverse modelview
    matrix when PathColorGenNV is called.

        xcoeffs[0..3]   = coeffs[0..3]   * MV^-1
        xcoeffs[4..7]   = coeffs[4..7]   * MV^-1
        xcoeffs[8..11]  = coeffs[8..11]  * MV^-1
        xcoeffs[12..15] = coeffs[12..12] * MV^-1

    [[ NOTATION:

       xxx[0..3] is a vector form from xxx[0], xxx[1], xxx[2], and xxx[3]

       MV^-1 is the inverse of the current modelview matrix when PathColorGenNV happens.

    ]]

    If the /genMode/ is NONE, no values from the /coeffs/ array are
    accessed and the R, G, B, and A components of a covered fragment's
    varying color (be it primary or secondary depending on color)
    are computed:

        R = Rc
        G = Gc
        B = Bc
        A = Ac

    If /colorFormat/ is LUMINANCE and /genMode/ is either OBJECT_LINEAR
    or PATH_OBJECT_BOUNDING_BOX_NV, then 3 values are accessed from the
    /coeffs/ array and the R, G, B, and A components of a covered
    fragment's varying color are computed:

        R = coeffs[0] * x + coeffs[1] * y + coeffs[2]
        G = coeffs[0] * x + coeffs[1] * y + coeffs[2]
        B = coeffs[0] * x + coeffs[1] * y + coeffs[2]
        A = Ac

    Alternatively if the /genMode/ is EYE_LINEAR, then 4 values are
    accessed and the varying color components are computed:

        R = xcoeffs[0] * xe + xcoeffs[1] * ye + xcoeffs[2] * ze + xcoeffs[3] * we
        G = xcoeffs[0] * xe + xcoeffs[1] * ye + xcoeffs[2] * ze + xcoeffs[3] * we
        B = xcoeffs[0] * xe + xcoeffs[1] * ye + xcoeffs[2] * ze + xcoeffs[3] * we
        A = Ac

    Alternatively if the /genMode/ is CONSTANT, then:

        R = xcoeffs[0]
        G = xcoeffs[0]
        B = xcoeffs[0]
        A = Ac

    If /colorFormat/ is INTENSITY and /genMode/ is either OBJECT_LINEAR
    or PATH_OBJECT_BOUNDING_BOX_NV, then 3 values are accessed from
    the /coeffs/ array and the R, G, B, and A components of a covered
    fragment's varying color are computed:

        R = coeffs[0] * x + coeffs[1] * y + coeffs[2]
        G = coeffs[0] * x + coeffs[1] * y + coeffs[2]
        B = coeffs[0] * x + coeffs[1] * y + coeffs[2]
        A = coeffs[0] * x + coeffs[1] * y + coeffs[2]

    Alternatively if the /genMode/ is EYE_LINEAR, then 4 values are
    accessed and the varying color components are computed:

        R = xcoeffs[0] * xe + xcoeffs[1] * ye + xcoeffs[2] * ze + xcoeffs[3] * we
        G = xcoeffs[0] * xe + xcoeffs[1] * ye + xcoeffs[2] * ze + xcoeffs[3] * we
        B = xcoeffs[0] * xe + xcoeffs[1] * ye + xcoeffs[2] * ze + xcoeffs[3] * we
        A = xcoeffs[0] * xe + xcoeffs[1] * ye + xcoeffs[2] * ze + xcoeffs[3] * we

    Alternatively if the /genMode/ is CONSTANT, then:

        R = xcoeffs[0]
        G = xcoeffs[0]
        B = xcoeffs[0]
        A = xcoeffs[0]

    If /colorFormat/ is ALPHA and /genMode/ is either OBJECT_LINEAR
    or PATH_OBJECT_BOUNDING_BOX_NV, then 3 values are accessed from
    the /coeffs/ array and the R, G, B, and A components of a covered
    fragment's varying color are computed:

        R = Rc
        G = Gc
        B = Bc
        A = coeffs[0] * x + coeffs[1] * y + coeffs[2]

    Alternatively if the /genMode/ is EYE_LINEAR, then 4 values are
    accessed and the varying color components are computed:

        R = Rc
        G = Gc
        B = Bc
        A = xcoeffs[0] * xe + xcoeffs[1] * ye + xcoeffs[2] * ze + xcoeffs[3] * we

    Alternatively if the /genMode/ is CONSTANT, then:

        R = Rc
        G = Gc
        B = Bc
        A = xcoeffs[0]

    If /colorFormat/ is RGB and /genMode/ is either OBJECT_LINEAR or
    PATH_OBJECT_BOUNDING_BOX_NV, then 9 values are accessed from the
    /coeffs/ array and the R, G, B, and A components of a covered
    fragment's varying color are computed:

        R = coeffs[0] * x + coeffs[1] * y + coeffs[2]
        G = coeffs[3] * x + coeffs[4] * y + coeffs[5]
        B = coeffs[6] * x + coeffs[7] * y + coeffs[8]
        A = Ac

    Alternatively if the /genMode/ is EYE_LINEAR, then 12 values are
    accessed and the varying color components are computed:

        R = xcoeffs[0] * xe + xcoeffs[1] * ye + xcoeffs[2]  * ze + xcoeffs[3]  * we
        G = xcoeffs[4] * xe + xcoeffs[5] * ye + xcoeffs[6]  * ze + xcoeffs[7]  * we
        B = xcoeffs[8] * xe + xcoeffs[9] * ye + xcoeffs[10] * ze + xcoeffs[11] * we
        A = Ac

    Alternatively if the /genMode/ is CONSTANT, then:

        R = xcoeffs[0]
        G = xcoeffs[1]
        B = xcoeffs[2]
        A = Ac

    If /colorFormat/ is RGBA and /genMode/ is either OBJECT_LINEAR
    or PATH_OBJECT_BOUNDING_BOX_NV, then 12 values are accessed from
    the /coeffs/ array and the R, G, B, and A components of a covered
    fragment's varying color are computed:

        R = coeffs[0] * x + coeffs[1]  * y + coeffs[2]
        G = coeffs[3] * x + coeffs[4]  * y + coeffs[5]
        B = coeffs[6] * x + coeffs[7]  * y + coeffs[8]
        A = coeffs[9] * x + coeffs[10] * y + coeffs[11]

    Alternatively if the /genMode/ is EYE_LINEAR, then 12 values are
    accessed and the varying color components are computed:

        R = xcoeffs[0]  * xe + xcoeffs[1]  * ye + xcoeffs[2]  * ze + xcoeffs[3]  * we
        G = xcoeffs[4]  * xe + xcoeffs[5]  * ye + xcoeffs[6]  * ze + xcoeffs[7]  * we
        B = xcoeffs[8]  * xe + xcoeffs[9]  * ye + xcoeffs[10] * ze + xcoeffs[11] * we
        A = xcoeffs[12] * xe + xcoeffs[13] * ye + xcoeffs[14] * ze + xcoeffs[15] * we

    Alternatively if the /genMode/ is CONSTANT, then:

        R = xcoeffs[0]
        G = xcoeffs[1]
        B = xcoeffs[2]
        A = xcoeffs[3]

    The state required for path color generation for each color (primary
    and secondary) is a four-valued integer for the path color generation
    mode and 16 floating-point coefficients.  The initial mode is NONE
    and the coefficients are all initially zero.

    As many coefficients are copied by the PathColorGenNV command
    to the 16 floating-point coefficient state as are referenced by
    the respective generation expression involving /colorFormat/ and
    /genMode/; unreferenced coefficients in the array of 16 coefficients
    are set to zero.

    TEXTURE COORDINATE SET GENERATION FOR PATH COVER COMMANDS

    The command

        void PathTexGenNV(enum texCoordSet,
                          enum genMode,
                          int components, const float *coeffs);

    controls how texture coordinate sets are computed for fragment
    shading operations that occur as a result of CoverFillPathNV or
    CoverStrokePathNV.

    /texCoordSet/ must be one of TEXTURE0 through
    TEXTUREn where /n/ is one less than the implementation-dependent
    value of MAX_TEXTURE_COORDS; otherwise INVALID_ENUM is generated.

    /genMode/ must be one of NONE, OBJECT_LINEAR,
    PATH_OBJECT_BOUNDING_BOX_NV, or EYE_LINEAR; otherwise INVALID_ENUM
    is generated.

    /components/ must be 0 if /genMode/ is NONE or for other allowed
    /genMode/ values must be one of 1, 2, 3, or 4; otherwise INVALID_VALUE
    is generated.  /components/ determines how many texture coordinate
    components of the texture coordinate set, how many coefficients read
    from the /coeffs/ array, and the linear equations used to generate the
    s, t, r, and q texture coordinates of the varying texture coordinate
    set specified by /texCoordSet/.

    In the following equations, coeffs[i] is the /i/th element (base
    zero) of the /coeffs/ array; sc, tc, rc, and qa are the s, t, r,
    and q texture coordinates of the texture coordinate set indicated
    by /texCoordSet/ when the path is covered; and x, y, z, and w are
    determined by the /genMode/ in the same manner as PathColorGenNV's
    /genMode/.

    When /genMode/ is EYE_LINEAR, xcoeffs[i] is the /i/th element (base
    zero) of a /xcoeffs/ array generated by multiplying each respective
    vector of four elements of coeffs by the current inverse modelview
    matrix when PathColorGenNV is called.

        xcoeffs[0..3]   = coeffs[0..3]   * MV^-1
        xcoeffs[4..7]   = coeffs[4..7]   * MV^-1
        xcoeffs[8..11]  = coeffs[8..11]  * MV^-1
        xcoeffs[12..15] = coeffs[12..12] * MV^-1

    [[ NOTATION:

       xxx[0..3] is a vector form from xxx[0], xxx[1], xxx[2], and xxx[3]

       MV^-1 is the inverse of the current modelview matrix when PathColorGenNV happens.

    ]]

    If the /components/ is 0, no values from the /coeffs/ array are
    accessed and the s, t, r, and q coordinates of a covered fragment's
    varying texture coordinate set for /texCoordSet/ are computed:

        s = sc
        t = tc
        r = rc
        q = qc

    If the /components/ is 1 and /genMode/ is either OBJECT_LINEAR or
    PATH_OBJECT_BOUNDING_BOX_NV, 3 values from the /coeffs/ array are
    accessed and the s, t, r, and q coordinates of a covered fragment's
    varying texture coordinate set for /texCoordSet/ are computed:

        s = coeffs[0] * x + coeffs[1] * y + coeffs[2]
        t = tc
        r = rc
        q = qc

    Alternatively if the /genMode/ is EYE_LINEAR, then 4 values are
    accessed and the varying texture coordinate set for /texunit/ are
    computed:

        s = xcoeffs[0] * xe + xcoeffs[1] * ye + xcoeffs[2] * ze + xcoeffs[3] * we
        t = tc
        r = rc
        q = qc

    Alternatively if the /genMode/ is CONSTANT, then:

        s = xcoeffs[0]
        t = tc
        r = rc
        q = qc

    If the /components/ is 2 and /genMode/ is either OBJECT_LINEAR or
    PATH_OBJECT_BOUNDING_BOX_NV, 6 values from the /coeffs/ array are accessed and the
    s, t, r, and q coordinates of a covered fragment's varying texture
    coordinate set for /texCoordSet/ are computed:

        s = coeffs[0] * x + coeffs[1] * y + coeffs[2]
        t = coeffs[3] * x + coeffs[4] * y + coeffs[5]
        r = rc
        q = qc

    Alternatively if the /genMode/ is EYE_LINEAR, then 8 values are
    accessed and the varying texture coordinate set for /texunit/ are
    computed:

        s = xcoeffs[0] * xe + xcoeffs[1] * ye + xcoeffs[2] * ze + xcoeffs[3] * we
        t = xcoeffs[4] * xe + xcoeffs[5] * ye + xcoeffs[6] * ze + xcoeffs[7] * we
        r = rc
        q = qc

    Alternatively if the /genMode/ is CONSTANT, then:

        s = xcoeffs[0]
        t = xcoeffs[1]
        r = rc
        q = qc

    If the /components/ is 3 and /genMode/ is either OBJECT_LINEAR or
    PATH_OBJECT_BOUNDING_BOX_NV, 9 values from the /coeffs/ array are accessed and the
    s, t, r, and q coordinates of a covered fragment's varying texture
    coordinate set for /texCoordSet/ are computed:

        s = coeffs[0] * x + coeffs[1] * y + coeffs[2]
        t = coeffs[3] * x + coeffs[4] * y + coeffs[5]
        r = coeffs[6] * x + coeffs[7] * y + coeffs[8]
        q = qc

    Alternatively if the /genMode/ is CONSTANT, then:

        s = xcoeffs[0]
        t = xcoeffs[1]
        r = xcoeffs[2]
        q = qc

    Alternatively if the /genMode/ is EYE_LINEAR, then 12 values are
    accessed and the varying texture coordinate set for /texunit/ are
    computed:

        s = xcoeffs[0] * xe + xcoeffs[1] * ye + xcoeffs[2]  * ze + xcoeffs[3]  * we
        t = xcoeffs[4] * xe + xcoeffs[5] * ye + xcoeffs[6]  * ze + xcoeffs[7]  * we
        r = xcoeffs[8] * xe + xcoeffs[9] * ye + xcoeffs[10] * ze + xcoeffs[11] * we
        q = qc

    If the /components/ is 4 and /genMode/ is either OBJECT_LINEAR or
    PATH_OBJECT_BOUNDING_BOX_NV, 12 values from the /coeffs/ array are accessed and the
    s, t, r, and q coordinates of a covered fragment's varying texture
    coordinate set for /texCoordSet/ are computed:

        s = coeffs[0] * x + coeffs[1]  * y + coeffs[2]
        t = coeffs[3] * x + coeffs[4]  * y + coeffs[5]
        r = coeffs[6] * x + coeffs[7]  * y + coeffs[8]
        q = coeffs[9] * x + coeffs[10] * y + coeffs[11]

    Alternatively if the /genMode/ is EYE_LINEAR, then 16 values are
    accessed and the varying texture coordinate set for /texunit/ are
    computed:

        s = xcoeffs[0]  * xe + xcoeffs[1]  * ye + xcoeffs[2]  * ze + xcoeffs[3]  * we
        t = xcoeffs[4]  * xe + xcoeffs[5]  * ye + xcoeffs[6]  * ze + xcoeffs[7]  * we
        r = xcoeffs[8]  * xe + xcoeffs[9]  * ye + xcoeffs[10] * ze + xcoeffs[11] * we
        q = xcoeffs[12] * xe + xcoeffs[13] * ye + xcoeffs[14] * ze + xcoeffs[15] * we

    Alternatively if the /genMode/ is CONSTANT, then:

        s = xcoeffs[0]
        t = xcoeffs[1]
        r = xcoeffs[2]
        q = xcoeffs[3]

    The state required for path color generation for each texture
    coordinate set is a four-valued integer for the path texture
    coordinate set generation mode and 16 floating-point coefficients.
    The initial mode is NONE and the coefficients are all initially zero.

    As many coefficients are copied by the PathTexGenNV command to
    the 16 floating-point coefficient state as are referenced by the
    respective generation expression involving /components/ and /genMode/;
    unreferenced coefficients in the array of 16 coefficients are set
    to zero.

    FOG COORDINATE GENERATION FOR PATH COVER COMMANDS

    The command

        void PathFogGenNV(enum genMode);

    controls how the fog coordinate is computed for fragment
    shading operations that occur as a result of CoverFillPathNV or
    CoverStrokePathNV.

    /genMode/ must be either FOG_COORDINATE or FRAGMENT_DEPTH; otherwise
    INVALID_ENUM is generated.

    If the /genMode/ is FOG_COORDINATE, then current fog coordinate is
    used (without varying) for all fragment generated by covering the
    filled or stroked path.

    If the /genMode/ is FRAGMENT_DEPTH, then the current fog coordinate
    is -ze, the interpolated negated (non-perspective-divided) eye-space
    Z coordinate from transforming of path's 2D coordinates transformed
    in eye space, so (xe, ye, ze, we) from section 2.12 ("Fixed-Function
    Vertex Transformation").

    The state required for path fog generation is a two-valued integer for
    the path fog generation mode; the mode is initially FRAGMENT_DEPTH.

    5.X.2.3 Instanced Path Stenciling and Covering

    Path rendering often depends on rendering a collection of paths at
    once. The most common case of this is rendering text as a set of
    glyphs corresponding to each character of text.  To support this
    usage efficiently, GL includes commands for instanced path stenciling
    and covering.

    The command

        void StencilFillPathInstancedNV(sizei numPaths,
                                        enum pathNameType, const void *paths,
                                        uint pathBase,
                                        enum fillMode, uint mask,
                                        enum transformType,
                                        const float *transformValues);

    stencils a sequence of filled paths.

    The /pathBase/ is an offset added to the /numPaths/ path names read
    from the /paths/ array (interpreted based on /pathNameType/).

    The /pathNameType/ determines the type of elements of the /paths/
    array and must be one of BYTE, UNSIGNED_BYTE, SHORT, UNSIGNED_SHORT,
    INT, UNSIGNED_INT, FLOAT, UTF8_NV, UTF16_NV, 2_BYTES, 3_BYTES,
    or 4_BYTES; otherwise the INVALID_ENUM error is generated.

    The /transformType/ must be one of NONE, TRANSLATE_X_NV,
    TRANSLATE_Y_NV, TRANSLATE_2D_NV, TRANSLATE_3D_NV, AFFINE_2D_NV,
    AFFINE_3D_NV, TRANSPOSE_AFFINE_2D_NV, or TRANSPOSE_AFFINE_3D_NV;
    otherwise the INVALID_ENUM error is generated.

    The /fillMode/ and /mask/ are validated identically to the same-named
    parameters of StencilFillPathNV.

    The StencilFillPathInstancedNV command is equivalent to:

        const float *v = transformValues;
        for (int i = 0; i<numPaths; i++) {
          double m[16];

          GetDoublev(MODELVIEW_MATRIX, m);  // save matrix
          v = applyTransformType(transformType, v);
          uint pathName;
          bool ok = getPathName(pathNameType, paths, pathBase, pathName);
          if (!ok)
            return;  // stop early
          if (IsPathNV(pathName)) {
            StencilFillPathNV(pathName, fillMode, mask);
          }
          MatrixLoaddEXT(MODELVIEW, m);  // restore matrix
        }

    assuming these helper functions for applyTransformType and
    getPathName:

        const float *applyTransformType(enum transformType, const float *v)
        {
          float m[16];
          switch (transformType) {
          case NONE:
            break;
          case TRANSLATE_X_NV:
            MatrixTranslateEXT(MODELVIEW, *v++, 0, 0);
            break;
          case TRANSLATE_Y_NV:
            MatrixTranslateEXT(MODELVIEW, 0, *v++, 0);
            break;
          case TRANSLATE_2D_NV:
            MatrixTranslateEXT(MODELVIEW, *v++, *v++, 0);
            break;
          case TRANSLATE_3D_NV:
            MatrixTranslateEXT(MODELVIEW, *v++, *v++, *v++);
            break;
          case AFFINE_2D_NV:
            m[0] =v[0]; m[4] =v[2]; m[8] =0; m[12]=v[4];
            m[1] =v[1]; m[5] =v[3]; m[9] =0; m[13]=v[5];
            m[2] =0;    m[6] =0;    m[10]=1; m[14]=0;
            m[3] =0;    m[7] =0;    m[11]=0; m[15]=1;
            v += 6;
            MatrixMultfEXT(MODELVIEW, m);
            break;
          case TRANSPOSE_AFFINE_2D_NV:
            m[0] =v[0]; m[4] =v[1]; m[8] =0; m[12]=v[2];
            m[1] =v[3]; m[5] =v[4]; m[9] =0; m[13]=v[5];
            m[2] =0;    m[6] =0;    m[10]=1; m[14]=0;
            m[3] =0;    m[7] =0;    m[11]=0; m[15]=1;
            v += 6;
            MatrixMultfEXT(MODELVIEW, m);
            break;
          case AFFINE_3D_NV:
            m[0] =v[0]; m[4] =v[3]; m[8] =v[6]; m[12]=v[9];
            m[1] =v[1]; m[5] =v[4]; m[9] =v[7]; m[13]=v[10];
            m[2] =v[2]; m[6] =v[5]; m[10]=v[8]; m[14]=v[11];
            m[3] =0;    m[7] =0;    m[11]=1;    m[15]=0;
            v += 12;
            MatrixMultfEXT(MODELVIEW, m);
            break;
          case TRANSPOSE_AFFINE_3D_NV:
            m[0] =v[0]; m[4] =v[1]; m[8] =v[2];  m[12]=v[3];
            m[1] =v[4]; m[5] =v[5]; m[9] =v[6];  m[13]=v[7];
            m[2] =v[8]; m[6] =v[9]; m[10]=v[10]; m[14]=v[11];
            m[3] =0;    m[7] =0;    m[11]=1;     m[15]=0;
            v += 12;
            MatrixMultfEXT(MODELVIEW, m);
            break;
          default:  // generate INVALID_ENUM
          }
          return v;
        }

        bool getPathName(enum pathNameType, const void *&paths,
                         uint pathBase, uint &pathName)
        {
          switch (pathNameType) {
          case BYTE:
            {
              const byte *p = (const byte*)paths;
              pathName = pathBase + p[0];
              paths = p+1;
              return true;
            }
          case UNSIGNED_BYTE:
            {
              const ubyte *p = (const ubyte*)paths;
              pathName = pathBase + p[0];
              paths = p+1;
              return true;
            }
          case SHORT:
            {
              const short *p = (const short*)paths;
              pathName = pathBase + p[0];
              paths = p+1;
              return true;
            }
          case UNSIGNED_SHORT:
            {
              const ushort *p = (const ushort*)paths;
              pathName = pathBase + p[0];
              paths = p+1;
              return true;
            }
          case INT:
            {
              const int *p = (const int*)paths;
              pathName = pathBase + p[0];
              paths = p+1;
              return true;
            }
          case UNSIGNED_INT:
            {
              const uint *p = (const uint*)paths;
              pathName = pathBase + p[0];
              paths = p+1;
              return true;
            }
          case FLOAT:
            {
              const float *p = (const float*)paths;
              pathName = pathBase + p[0];
              paths = p+1;
              return true;
            }
          case 2_BYTES:
            {
              const ubyte *p = (const ubyte*)paths;
              pathName = pathBase + (p[0]<<8 | p[1]);
              paths = p+2;
              return true;
            }
          case 3_BYTES:
            {
              const ubyte *p = (const ubyte*)paths;
              pathName = pathBase + (p[0]<<16 | p[1]<<8 | p[0]);
              paths = p+3;
              return true;
            }
          case 4_BYTES:
            {
              const ubyte *p = (const ubyte*)paths;
              pathName = pathBase + (p[0]<<24 | p[1]<<16 | p[2]<<8 | p[3]);
              paths = p+4;
              return true;
            }
          case UTF8_NV:
            {
              const ubyte *p = (const ubyte*)paths;
              ubyte c0 = p[0];
              if ((c0 & 0x80) == 0x00) {
                // Zero continuation (0 to 127)
                pathName = pathBase + c0;
                p += 1;
              } else {
                ubyte c1 = p[1];
                if ((c1 & 0xC0) != 0x80) {
                  // Stop processing the UTF byte sequence early.
                  return false;
                }
                if ((c0 & 0xE0) == 0xC0) {
                  // One contination (128 to 2047)
                  pathName = pathBase + ((c1 & 0x3F) | (c0 & 0x1F) << 6);
                  if (pathName < 128) {
                    return false;
                  }
                  p += 2;
                } else {
                  ubyte c2 = p[2];
                  if ((c2 & 0xC0) != 0x80) {
                    // Stop processing the UTF byte sequence early.
                    return false;
                  }
                  if ((c0 & 0xF0) == 0xE0) {
                    // Two continuation (2048 to 55295 and 57344 to 65535)
                    pathName = pathBase + ((c2 & 0x3F) | (c1 & 0x3F) << 6 |
                                           (c0 & 0xF) << 12);
                    if ((pathName >= 55296) && (pathName <= 57343)) {
                      // Stop processing the UTF byte sequence early.
                      return false;
                    }
                    if (pathName < 2048) {
                      return false;
                    }
                    p += 3;
                  } else {
                    ubyte c3 = p[3];
                    if ((c3 & 0xC0) != 0x80) {
                      // Stop processing the UTF byte sequence early.
                      return false;
                    }
                    if ((c0 & 0xF8) == 0xF0) {
                      // Three continuation (65536 to 1114111)
                      pathName = pathBase + ((c3 & 0x3F) | (c2 & 0x3F) << 6 |
                                             (c1 & 0x3F) << 12 | (c0 & 0x7) << 18);
                      if (pathName < 65536 && pathName > 1114111) {
                        return false;
                      }
                      p += 4;
                    } else {
                      // Skip invalid or restricted encodings.
                      // Stop processing the UTF byte sequence early.
                      return false;
                    }
                  }
                }
              }
              paths = p;
              return true;
            }
          case UTF16_NV:
            {
              const ushort *p = (const ushort*)paths;

              ushort s0 = p[0];
              if ((s0 < 0xDB00) || (s0 > 0xDFFF)) {
                  pathName = pathBase + s0;
                  p += 1;
              } else {
                if ((s0 >= 0xDB00) && (s0 <= 0xDBFF)) {
                  ushort s1 = p[1];
                  if ((s1 >= 0xDC00) && (s1 <= 0xDFFF)) {
                    pathName = pathBase + (((s0 & 0x3FF) << 10 |
                                            (s1 & 0x3FF)) + 0x10000);
                    p += 2;
                  } else {
                    // Stop processing the UTF byte sequence early.
                    return false;
                  }
                } else {
                  return false;
                }
              }
              paths = p;
              return true;
            }
          default:
              << generate INVALID_ENUM >>
              return false;
          }
        }

    The command

        void StencilStrokePathInstancedNV(sizei numPaths,
                                          enum pathNameType, const void *paths,
                                          uint pathBase,
                                          int reference, uint mask,
                                          enum transformType,
                                          const float *transformValues);

    stencils a sequence of stroked paths and is equivalent to:

        const float *v = transformValues;
        for (int i = 0; i<numPaths; i++) {
          double m[16];

          GetDoublev(MODELVIEW_MATRIX, m);  // save matrix
          v = applyTransformType(transformType, v);
          uint pathName;
          bool ok = getPathName(pathNameType, paths, pathBase, pathName);
          if (!ok)
            return;  // stop early
          if (IsPathNV(pathName)) {
            StencilStrokePathNV(pathName, reference, mask);
          }
          MatrixLoaddEXT(MODELVIEW, m);  // restore matrix
        }

    assume the helper functions for applyTransformType and
    getPathName defined above.

    The command

        void CoverFillPathInstancedNV(sizei numPaths,
                                      enum pathNameType, const void *paths,
                                      uint pathBase,
                                      enum coverMode,
                                      enum transformType,
                                      const float *transformValues);

    covers a sequence of filled paths and is equivalent to:

        if (coverMode == BOUNDING_BOX_OF_BOUNDING_BOXES_NV) {
          renderBoundingBox(PATH_FILL_BOUNDING_BOX_NV,
                            numPaths,
                            pathNameType, paths,
                            pathBase,
                            transformType, transformValues);
        } else {
          const float *v = transformValues;
          for (int i = 0; i<numPaths; i++) {
            double m[16];
            uint path;

            GetDoublev(MODELVIEW_MATRIX, m);  // save matrix
            v = applyTransformType(transformType, v);
            uint pathName;
            bool ok = getPathName(pathNameType, paths, pathBase, pathName);
            if (!ok)
              return;  // stop early
            if (IsPathNV(pathName)) {
              << set fragment shader instance ID to i >>
              CoverFillPathNV(pathName, cover);
            }
            MatrixLoaddEXT(MODELVIEW, m);  // restore matrix
          }
        }

    assuming these helper functions for applyTransformType and
    getPathName defined above as well as:

        void renderBoundingBox(enum boundingBoxType,
                               sizei numPaths,
                               enum pathNameType,
                               const void *paths,
                               uint pathBase,
                               enum transformType,
                               const float *transformValues)
        {
          boolean hasBounds = FALSE;
          float boundsUnion[4], bounds[4];

          const float *v = transformValues;
          for (int i = 0; i<numPaths; i++) {
            uint pathName;
            bool ok = getPathName(pathNameType, paths, pathBase, pathName);
            if (!ok)
              return;  // stop early
            if (IsPathNV(pathName)) {
              GetPathParameterfvNV(pathName, boundingBoxType, bounds);
              switch (transformType) {
              case NONE:
                break;
              case TRANSLATE_X_NV:
                bounds[0] += v[0];
                bounds[2] += v[0];
                v += 1;
                break;
              case TRANSLATE_Y_NV:
                bounds[1] += v[0];
                bounds[3] += v[0];
                v += 1;
                break;
              case TRANSLATE_2D_NV:
                bounds[0] += v[0];
                bounds[1] += v[1];
                bounds[2] += v[0];
                bounds[3] += v[1];
                v += 2;
                break;
              case TRANSLATE_3D_NV: // ignores v[2]
                bounds[0] += v[0];
                bounds[1] += v[1];
                bounds[2] += v[0];
                bounds[3] += v[1];
                v += 3;
                break;
              case AFFINE_2D_NV:
                bounds[0] = bounds[0]*v[0] + bounds[0]*v[2] + v[4];
                bounds[1] = bounds[1]*v[1] + bounds[1]*v[3] + v[5];
                bounds[2] = bounds[2]*v[0] + bounds[2]*v[2] + v[4];
                bounds[3] = bounds[3]*v[1] + bounds[3]*v[3] + v[5];
                v += 6;
                break;
              case TRANSPOSE_AFFINE_2D_NV:
                bounds[0] = bounds[0]*v[0] + bounds[0]*v[1] + v[2];
                bounds[1] = bounds[1]*v[3] + bounds[1]*v[4] + v[5];
                bounds[2] = bounds[2]*v[0] + bounds[2]*v[1] + v[2];
                bounds[3] = bounds[3]*v[3] + bounds[3]*v[4] + v[5];
                v += 6;
                break;
              case AFFINE_3D_NV:  // ignores v[2], v[5], v[6..8], v[11]
                bounds[0] = bounds[0]*v[0] + bounds[0]*v[3] + v[9];
                bounds[1] = bounds[1]*v[1] + bounds[1]*v[4] + v[10];
                bounds[2] = bounds[2]*v[0] + bounds[2]*v[3] + v[9];
                bounds[3] = bounds[3]*v[1] + bounds[3]*v[4] + v[10];
                v += 12;
                break;
              case TRANSPOSE_AFFINE_3D_NV:  // ignores v[2], v[6], v[8..11]
                bounds[0] = bounds[0]*v[0] + bounds[0]*v[1] + v[3];
                bounds[1] = bounds[1]*v[4] + bounds[1]*v[5] + v[7];
                bounds[2] = bounds[2]*v[0] + bounds[2]*v[1] + v[3];
                bounds[3] = bounds[3]*v[4] + bounds[3]*v[5] + v[7];
                v += 12;
                break;
              default:  // generate INVALID_ENUM
              }
              if (bounds[0] > bounds[2]) {
                float t = bounds[2];
                bounds[2] = bounds[0];
                bounds[0] = t;
              }
              if (bounds[1] > bounds[3]) {
                float t = bounds[3];
                bounds[3] = bounds[1];
                bounds[1] = t;
              }
              if (hasBounds) {
                if (bounds[0] < boundsUnion[0]) {
                  boundsUnion[0] = bounds[0];
                }
                if (bounds[1] < boundsUnion[1]) {
                  boundsUnion[1] = bounds[1];
                }
                if (bounds[2] > boundsUnion[2]) {
                  boundsUnion[2] = bounds[2];
                }
                if (bounds[3] > boundsUnion[3]) {
                  boundsUnion[3] = bounds[3];
                }
              } else {
                for (int i=0; i<4; i++) {
                  boundsUnion[i] = bounds[i];
                }
                hasBounds = TRUE;
              }
            }
          }
          if (hasBounds) {
            boolean polygonSmoothEnable = IsEnabled(POLYGON_SMOOTH);
            int polygonModes[2];
            GetIntegerv(POLYGON_MODE, polygonModes);
            PolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            Disable(GL_POLYGON_SMOOTH);
            Rectf(boundsUnion[0], boundsUnion[1], boundsUnion[2], boundsUnion[3]);
            PolygonMode(FRONT, polygonModes[0]);
            PolygonMode(BACK, polygonModes[1]);
            if (polygonSmoothEnable) {
              Enable(POLYGON_SMOOTH);
            } else {
              Disable(POLYGON_SMOOTH);
            }
          }
        }

    The GetPathParameterfvNV query, used in the code above, is introduced
    in section 6.X.1 ("Path Object Parameter Queries").

    The command

         void CoverStrokePathInstancedNV(sizei numPaths,
                                         enum pathNameType, const void *paths,
                                         uint pathBase,
                                         enum coverMode,
                                         enum transformType,
                                         const float *transformValues);

    covers a sequence of stroked paths and is equivalent to:

        if (coverage == BOUNDING_BOX_OF_BOUNDING_BOXES_NV) {
          renderBoundingBox(PATH_STROKE_BOUNDING_BOX_NV,
                            numPaths,
                            pathNameType, paths,
                            pathBase,
                            transformType, transformValues);
        } else {
          const float *v = transformValues;
          for (int i = 0; i<numPaths; i++) {
            double m[16];

            GetDoublev(MODELVIEW_MATRIX, m);  // save matrix
            v = applyTransformType(transformType, v);
            uint pathName;
            bool ok = getPathName(pathNameType, paths, pathBase, pathName);
            if (!ok)
              return;  // stop early
            if (IsPathNV(pathName)) {
              << set fragment shader instance ID to i >>
              CoverStrokePathNV(pathName, cover);
            }
            MatrixLoaddEXT(MODELVIEW, m);  // restore matrix
          }
        }

    assuming these helper functions for applyTransformType,
    getPathName, and renderBoundingBox defined above.

    5.X.2.4 Path Stenciling Then Covering

    The following command combine the stencil and cover operations on
    paths into a single command.

    The command

        void StencilThenCoverFillPathNV(uint path, enum fillMode, uint mask, enum coverMode);

    is equivalent to the two commands

        StencilFillPathNV(path, fillMode, mask);
        CoverFillPathNV(path, coverMode);

    unless either command would generate an error; for any such error
    other than OUT_OF_MEMORY, only that error is generated.

    The command

        void StencilThenCoverStrokePathNV(uint path, int reference, uint mask, enum coverMode);

    is equivalent to the two commands

        StencilStrokePathNV(uint path, int reference, uint mask);
        CoverStrokePathNV(uint path, enum coverMode);

    unless either command would generate an error; for any such error
    other than OUT_OF_MEMORY, only that error is generated.

    The command

        void StencilThenCoverFillPathInstancedNV(sizei numPaths,
                                                 enum pathNameType, const void *paths,
                                                 uint pathBase,
                                                 enum fillMode, uint mask,
                                                 enum coverMode,
                                                 enum transformType,
                                                 const float *transformValues);

    is equivalent to the two commands

        StencilFillPathInstancedNV(sizei numPaths,
                                   enum pathNameType, const void *paths,
                                   uint pathBase,
                                   enum fillMode, uint mask,
                                   enum coverMode,
                                   enum transformType,
                                   const float *transformValues);
        CoverFillPathInstancedNV(sizei numPaths,
                                 enum pathNameType, const void *paths,
                                 uint pathBase,
                                 enum fillMode, uint mask,
                                 enum coverMode,
                                 enum transformType,
                                 const float *transformValues);

    unless either command would generate an error; for any such error
    other than OUT_OF_MEMORY, only that error is generated.

    The command

        void StencilThenCoverStrokePathInstancedNV(sizei numPaths,
                                                   enum pathNameType, const void *paths,
                                                   uint pathBase,
                                                   int reference, uint mask,
                                                   enum coverMode,
                                                   enum transformType,
                                                   const float *transformValues);

    is equivalent to the two commands

        StencilStrokePathInstancedNV(sizei numPaths,
                                     enum pathNameType, const void *paths,
                                     uint pathBase,
                                     int reference, uint mask,
                                     enum transformType,
                                     const float *transformValues);
        CoverStrokePathInstancedNV(sizei numPaths,
                                   enum pathNameType, const void *paths,
                                   uint pathBase,
                                   enum coverMode,
                                   enum transformType,
                                   const float *transformValues);

    unless either command would generate an error; for any such error
    other than OUT_OF_MEMORY, only that error is generated.

 -- Section 5.4 "Display Lists"

    Add to the list of commands not compiled into display lists:

    "Path objects:  GenPathsNV, DeletePathsNV."

Additions to Chapter 6 of the OpenGL 3.2 (unabridged) Specification (State and
State Requests)

 -- Insert section 6.X "Path Object Queries" after 6.1.18 "Renderbuffer
    Object Queries"

    6.X. Path Rendering Queries

    6.X.1. Path Object Parameter Queries

    The queries

        void GetPathParameterivNV(uint path, enum pname, int *value);
        void GetPathParameterfvNV(uint path, enum pname, float *value);

    obtains the current value of the /param/ path parameter of the path
    object named /name/; the error INVALID_OPERATION is generated if
    /name/ is not an existing path object.  /value/ is a pointer to a
    scalar or array of the appropriate type, int for GetPathParameterivNV
    and float for GetPathParameterfvNV, in which to place the returned
    data.

    Table 6.readOnlyPathParameters

        Name                         Type     Description
        ---------------------------  -------  ----------------------------
        PATH_COMMAND_COUNT_NV        int      Length of the path's
                                              command sequence
        PATH_COORD_COUNT_NV          int      Length of the path's
                                              coordinate sequence
        PATH_DASH_ARRAY_COUNT_NV     int      Length of the path's
                                              dash array
        PATH_COMPUTED_LENGTH_NV      float    Computed path-space
                                              length of all the
                                              segments in the path
                                              (see section 6.X.4)
        PATH_OBJECT_BOUNDING_BOX_NV  4*float  tight path-space bounding
                                              box around the path's
                                              covered fill region
        PATH_FILL_BOUNDING_BOX_NV    4*float  Conservative path-space
                                              bounding box around the
                                              path's covered fill region
        PATH_STROKE_BOUNDING_BOX_NV  4*float  Conservative path-space
                                              bounding box around the
                                              path's covered stroke region

    /param/ must be one of the tokens listed in Table 5.pathParameters
    or Table 6.readOnlyPathParameters; otherwise the INVALID_ENUM
    error is generated.  The parameters from Table 5.pathParameters
    always return a single (scalar) value.  The parameters from
    Table 6.readOnlyPathParameters a single (scalar) value for all the
    parameters but the PATH_*_BOUNDING_BOX_NV parameters; these bounding
    box parameters return a vector of 4 values.  These four values are
    the minimum (x1,y1) corner of the respective path-space bounding
    box and the maximum (x2,y2) corner of the respective path-space
    orthogonally aligned bounding box, returned in (x1,y1,x2,y2) order.
    (This guarantees x1<=x2 and y1<=y2.)  Float parameters queried by
    GetPathParameterivNV are rounded to the nearest integer (where values
    with a floating-point fraction of 0.5 round up).

    The PATH_OBJECT_BOUNDING_BOX_NV bounding box is intended to bound
    tightly the region of path space containing the path's outline.
    The PATH_FILL_BOUNDING_BOX_NV bounding box matches the rectangle
    region covered by the CoverFillPathNV command with the BOUNDING_BOX_NV
    /coverMode/.  With either the PATH_OBJECT_BOUNDING_BOX_NV or
    PATH_FILL_BOUNDING_BOX_NV bounding boxes of a path object, a point at
    (x,y) such that x<x1 or x>x2 or y<y1 or y>y2 is guaranteed to /not/
    be within the filled outline of the path.

    The PATH_STROKE_BOUNDING_BOX_NV bounding box matches the rectangle
    region covered by the CoverFillPathNV command with the BOUNDING_BOX_NV
    /coverMode/.  With the PATH_STROKE_BOUNDING_BOX_NV bounding box of
    a path object, a point at (x,y) such that x<x1 or x>x2 or y<y1 or
    y>y2 is guaranteed to /not/ be within the stroked region of the path.

    6.X.2. Path Object Varying Arrays Queries

    Path objects support a variable number of commands, coordinates,
    and dash lengths.

    The query

        void GetPathCommandsNV(uint path, ubyte *commands);

    returns the sequence of commands within the path object named /name/
    into the array named /commands/; the error INVALID_OPERATION is
    generated if /name/ is not an existing path object.  The number of
    commands returned is identical to the value of the path object's
    PATH_COMMAND_COUNT_NV parameter.  The application is responsible
    for ensuring /commands/ array has sufficient space.

    Any path commands specified with a character alias value (from Table
    5.pathCommands) is returned as the command's token value instead.

    The query

        void GetPathCoordsNV(uint path, float *coords);

    returns the sequence of coordinates within the path object named
    /name/ into the array named /coords/; the error INVALID_OPERATION
    is generated if /name/ is not an existing path object.  The number
    of commands returned is identical to the value of the path object's
    PATH_COORD_COUNT_NV parameter.  The application is responsible for
    ensuring /coords/ array has sufficient space.

    Boolean coordinates such as the large/small and sweep flags for arcs
    are always returned as 1.0 or 0.0 for true and false respectively.
    Other coordinates are returned as they were specified.

    The query

        void GetPathDashArrayNV(uint path, float *dashArray);

    returns the sequence of dash lengths within the path object named
    /name/ into the array named /coords/; the error INVALID_OPERATION is
    generated if /name/ is not an existing path object.  The number of
    dash lengths returned is identical to the value of the path object's
    PATH_DASH_ARRAY_COUNT_NV parameter.  The application is responsible
    for ensuring /dashArray/ has sufficient space.

    6.X.3. Path Object Glyph Typographic Queries

    GLYPH METRIC QUERIES

    To facilitate proper text layout, the command

        void GetPathMetricsNV(bitfield metricQueryMask,
                              sizei numPaths,
                              enum pathNameType, const void *paths,
                              uint pathBase,
                              sizei stride,
                              float *metrics);

    queries glyph metrics associated with a sequence of path objects
    specified by the /glyphBase/, /count/, /pathNameType/, and /paths/
    parameters.  Metrics are associated with path objects specified by
    PathGlyphsNV or PathGlyphRangeNV (see section 5.X.1.3).

    There are two kinds of metrics:

    *  Per-glyph metrics that are typically different for each glyph.

    *  Per-font face metrics that are identical for all glyphs belonging
       to a given font face.

    Per-font face metrics are aggregate metrics such as the maximum
    ascender or descender for all the glyphs in the font face.

    /metricQueryMask/ is a bitfield constructed from the bits listed
    in Table 6.perGlyphMetrics and Table 6.perFontFaceMetrics.  If a bit
    is set in /metricQueryMask/ not listed in these tables, the error
    INVALID_VALUE is generated.

    /stride/ is the byte (machine units) offset separating each group of
    returned metrics for a given path object.  If /stride/ is negative
    or /stride/ is not a multiple of the size of float in bytes (machine
    units), the INVALID_VALUE error is generated.  The INVALID_OPERATION
    error is generated if /stride/ divided by the size of float in bytes
    is not either zero or else greater than or equal to the number of
    metrics specified for querying in the metricQueryMask (based on the
    number of specified bits specified in the mask) times the size of
    float in bytes.  A /stride/ of zero is specially handled; the value
    zero is interpreted to indicate the number of bytes (machine units)
    such that the all the metrics are written in a tightly packed array,
    so the size of float in bytes times the number of specified bits in
    the /metricQueryMask/ bitfield.

    For path objects not created with either PathGlyphsNV or
    PathGlyphRangeNV or non-existent, all glyph metrics return -1.

    This metric information for a path object is /not/ updated if
    the commands or coordinates or parameters of that path object are
    changed.

    Figure 6.horizontalGlyphMetrics:  Horizontal Glyph Metrics

                ^
                |    xMin         xMax
                |     |            |
                |     |   width    |
                |     |<---------->|
                |     |            |
                |     +============+ - - - - - - - - - - - yMax
                |     I            I   ^               ^
                |     I            I   | hBearingY     |
                |     I            I   |               |
      hBearingX |---->I  GLYPH     I   |        height |
                |     I   OUTLINE  I   |               |
            ----O-----I------------I------*--->        |
               /|     I    HERE    I      |            |
              / |     I            I      |            v
        origin  |     +============+ - - -|- - - - - - - - yMin
                |                         |
                |------------------------>|
                |   hAdvance              |

    Figure 6.verticalGlyphMetrics:  Vertical Glyph Metrics

               vBearingX
              |<---------|   origin
              |          | /
              |          |/
    ---------------------O----------------------------->
              |          |                  |    |
              |          |       vBearingY  |    |
              |          |                  v    |
     yMax - - +================+ - - - - - - - - |
              I          |     I     ^           |
              I          |     I     |           |
              I    GLYPH |     I     |           |
              I     OUTLINE    I     | height    |
              I      HERE|     I     |           |
              I          |     I     |           |
              I          |     I     |           |
              I          |     I     v           | vAdvance
     yMin - - +================+ - - -           |
              |          |     |                 v
              |          * - - - - - - - - - - - -
              |          |     |
             xMin        v   xMax

    Table 6.perGlyphMetrics

                                                             Bit number
                                                 Glyph       from LSB
        Bit field name                           metric tag  in bitmask  Description (units in path space)
        ---------------------------------------  ----------  ----------  -------------------------------------------
        GLYPH_WIDTH_BIT_NV                       width       0           Glyph's width
        GLYPH_HEIGHT_BIT_NV                      height      1           Glyph's height
        GLYPH_HORIZONTAL_BEARING_X_BIT_NV        hBearingX   2           Left side bearing for horizontal layout
        GLYPH_HORIZONTAL_BEARING_Y_BIT_NV        hBearingY   3           Top side bearing for horizontal layout
        GLYPH_HORIZONTAL_BEARING_ADVANCE_BIT_NV  hAdvance    4           Advance width for horizontal layout
        GLYPH_VERTICAL_BEARING_X_BIT_NV          vBearingX   5           Left side bearing for vertical layout
        GLYPH_VERTICAL_BEARING_Y_BIT_NV          vBearingY   6           Top side bearing for vertical layout
        GLYPH_VERTICAL_BEARING_ADVANCE_BIT_NV    vAdvance    7           Advance height for vertical layout
        GLYPH_HAS_KERNING_BIT_NV                 -           8           True if glyph has a kerning table.

    Table 6.perFontFaceMetrics

                                                 Bit number
                                                 from LSB
        Bit field name                           in bitmask  Description (units in path space)
        ---------------------------------------  ----------  ---------------------------------------------------
        FONT_X_MIN_BOUNDS_BIT_NV                 16          Horizontal minimum (left-most) of the font bounding
                                                             box.  The font bounding box (this metric and the
                                                             next 3) is large enough to contain any glyph from
                                                             the font face.
        FONT_Y_MIN_BOUNDS_BIT_NV                 17          Vertical minimum (bottom-most) of the font bounding
                                                             box.
        FONT_X_MAX_BOUNDS_BIT_NV                 18          Horizontal maximum (right-most) of the font
                                                             bounding box.
        FONT_Y_MAX_BOUNDS_BIT_NV                 19          Vertical maximum (top-most) of the font bounding
                                                             box.
        FONT_UNITS_PER_EM_BIT_NV                 20          Number of units in path space (font units) per
                                                             Em square for this font face.  This is typically
                                                             2048 for TrueType fonts, and 1000 for PostScript
                                                             fonts.
        FONT_ASCENDER_BIT_NV                     21          Typographic ascender of the font face.  For font
                                                             formats not supplying this information, this value
                                                             is the same as FONT_Y_MAX_BOUNDS_BIT_NV.
        FONT_DESCENDER_BIT_NV                    22          Typographic descender of the font face (always a
                                                             positive value).  For font formats not supplying
                                                             this information, this value is the same as
                                                             FONT_Y_MIN_BOUNDS_BIT_NV.
        FONT_HEIGHT_BIT_NV                       23          Vertical distance between two consecutive baselines
                                                             in the font face (always a positive value).
        FONT_MAX_ADVANCE_WIDTH_BIT_NV            24          Maximal advance width for all glyphs in this font
                                                             face.  (Intended to make word wrapping computations
                                                             easier.)
        FONT_MAX_ADVANCE_HEIGHT_BIT_NV           25          Maximal advance height for all glyphs in this
                                                             font face for vertical layout.  For font formats
                                                             not supplying this information, this value is the
                                                             same as FONT_HEIGHT_BIT_NV.
        FONT_UNDERLINE_POSITION_BIT_NV           26          Position of the underline line for this font face.
                                                             This position is the center of the underling stem.
        FONT_UNDERLINE_THICKNESS_BIT_NV          27          Thickness of the underline of this font face.
        FONT_HAS_KERNING_BIT_NV                  28          True if font face provides a kerning table
        FONT_NUM_GLYPH_INDICES_BIT_NV            29          Number of glyph indices for this font.

    consulted by the GetPathSpacingNV command discussed below
    ("GLYPH SPACING QUERIES").

    The query

        void GetPathMetricRangeNV(bitfield metricQueryMask,
                                  uint firstPathName,
                                  sizei numPaths,
                                  sizei stride,
                                  float *metrics);

    is equivalent to

        int *array = malloc(sizeof(int)*numGlyphs);
        if (array) {
          for (int i=0; i<numGlyphs; i++) {
            array[i] = i + firstPathName;
          }
          GetPathMetricsNV(metricQueryMask,
                           numPaths,
                           INT, array,
                           pathBase,
                           stride, metrics);
          free(array);
        } else {
          // generate OUT_OF_MEMORY error
        }

    GLYPH SPACING QUERIES

    The query

        void GetPathSpacingNV(enum pathListMode,
                              sizei numPaths,
                              enum pathNameType, const void *paths,
                              uint pathBase,
                              float advanceScale,
                              float kerningScale,
                              enum transformType,
                              float *returnedSpacing);

    returns a sequence of /numPaths/-1 glyph spacing vectors in path
    space for spacing the specified sequence of path object pairs.
    The returned vectors are written into the /returnedSpacing/ array.

    /pathListMode/ must be one of ADJACENT_PAIRS_NV,
    ACCUM_ADJACENT_PAIRS_NV, or FIRST_TO_REST_NV; otherwise the
    INVALID_ENUM error is generated.

    If /numPaths/ is negative, the error INVALID_VALUE is generated

    /pathNameType/ determines the type of elements of the /paths/ array
    and must be one of BYTE, UNSIGNED_BYTE, SHORT, UNSIGNED_SHORT, INT,
    UNSIGNED_INT, FLOAT, UTF8_NV, UTF16_NV, 2_BYTES, 3_BYTES, or 4_BYTES;
    otherwise the INVALID_ENUM error is generated.

    /transformType/ must be either TRANSLATE_X_NV or TRANSLATE_2D_NV;
    otherwise the INVALID_ENUM error is generated.

    In the absence of parameter errors, the following pseudo-code
    implements this query:

        double accumX = 0,
               accumY = 0;
        float returnX = 0,
              returnY = 0;
        uint firstPath;
        bool ok = getPathName(pathNameType, paths, pathBase, firstPath);
        if (!ok)
          return;  // stop early
        for (int i = 0; i<numPaths-1; i++) {
          uint secondPath;
          bool ok = getPathName(pathNameType, paths, pathBase, secondPath);
          if (!ok)
            return;  // stop early
          if (transformType == TRANSLATE_X_NV) {
            returnedSpacing[i] = returnX;
          } else {
            // transformType == TRANSLATE_2D_NV
            returnedSpacing[2*i  ] = returnX;
            returnedSpacing[2*i+1] = returnY;
          }
          float x = advanceScale * advanceX(firstPath) +
                    kerningScale * kerningX(firstPath,secondPath);
          float y = kerningScale * kerningY(firstPath,secondPath);
          if (pathListMode == ACCUM_ADJACENT_PAIRS_NV) {
            returnX = accumX;
            returnY = accumY;
            accumX += x;
            accumY += y;
          } else {
            returnX = x;
            returnY = y;
          }
          if (pathListMode != FIRST_TO_REST_NV) {
            firstPath = secondPath;
          }
        }

    The getPathName function is found in section 5.X.2.3 (Instanced Path
    Stenciling and Covering).

    The advance, kerningX, and kerningY functions operate as follows:

    The advance function returns the hAdvance metric of path object
    name passed to the function, but if the path object lacks glyph
    metrics, the difference between the horizontal bounds of the path
    object's bounding box (determined by PATH_OBJECT_BOUNDING_BOX_NV)
    are returned instead.  If the named path object passed to advance
    does not exist, zero is returned.

    The kerningX and kerningY functions return the X and Y kerning
    distances respectively between the character codes of the first
    (typically left) and second (typically right) path objects if they
    belong to the same font face; otherwise, zero is returned.  If the
    vertical kerning metrics are unavailable for the font face or either
    named path object does not exist, zero is returned for kerningY.
    (When the FONT_HAS_KERNING_BIT_NV or GLYPH_HAS_KERNING_BIT_NV glyph
    metrics are false for the first path object name, kerningX and
    kerningY must return zero.)

    Spacing information (horizontal advance or kerning) in a path object
    is not changed if the path's commands, coordinates, or parameters
    change--except in the case where the horizontal kerning value is
    determined by the first path's object bounding box.

    6.X.4. Path Object Geometric Queries

    The query

        boolean IsPointInFillPathNV(uint path,
                                    uint mask, float x, float y);

    computes the winding number of the path-space 2D (x,y) point given
    by /x/ and /y/ with respect to the path object named /path/ and
    returns TRUE if the winding count ANDed with /mask/ is non-zero;
    otherwise the query returns FALSE.  If the /mask/ parameter is zero,
    substitute the path object's PATH_FILL_MASK_NV parameter value
    when ANDing with the winding count.  The error INVALID_OPERATION is
    generated if /path/ does not exist.

    This point-inside computation occurs in /path space/ rather than in
    the window space as the StencilFillPathNV command operates.

    The query

        boolean IsPointInStrokePathNV(uint path,
                                      float x, float y);

    returns TRUE if the path-space 2D (x,y) point given by /x/ and
    /y/ is within the stroked region of the path object named /path/;
    otherwise the query returns FALSE.  The error INVALID_OPERATION is
    generated if /path/ does not exist.

    The stroked region's stroke width is specified by the path object's
    stroke width parameter.

    The stroked region is defined as in section 5.X.2.1 ("Path
    Stenciling") so accounts for the path object's current end cap,
    join style, and dashing parameters.

    This point-inside computation occurs in /path space/ rather than in
    the window space as the StencilStrokePathNV command operates.

    The query

        float GetPathLengthNV(uint path,
                              sizei startSegment, sizei numSegments);

    returns an approximation of the geometric length of a given portion
    of a path object named /path/.  The portion of the path measured is
    from the (0-indexed) /startSegment/ through the next /numSegments/.
    The returned length is measured in path-space units.  The error
    INVALID_OPERATION is generated if /path/ does not exist.

    The geometric length of the path's measured portion depends only
    on the path's commands and associated coordinates for the indicated
    range of segments and the respective coordinates of these segments.
    The geometric length of the path does not, for example, depend on
    the path's dashing parameters.

    The MOVE_TO_NV and RELATIVE_MOVE_TO_NV commands contribute zero
    units to the computed geometric length.  For all other path commands,
    a path segment's geometric length contribution /s/ is:

        s = int(sqrt(fx(t)^2+fy(t)^2), t, 0, 1)

    [[ int(f(t),t,a,b) computes the definite integration of the function
       f(t) over the interval [a,b]. ]]

    where /fx/ and /fy/ is the partial derivative of the command's
    respective path segment parametric function found in Table
    5.pathCommands.

    The return value, assuming no error, is the sum of all /s/ values
    for segments /startSegment/ through /startSegment/+/numSegments/-1
    inclusive.  If /numSegments/ is zero and no error is generated,
    0 is returned.

    The INVALID_VALUE error is generated in any of the following
    circumstances:

        *   /startSegment/ is negative,

        *   /numSegments/ is negative,

        *   /startSegment/+/numSegments/-1 is greater than the index of
            the final path segment.

    If an error occurs, -1.0 is returned.  When no error occurs, the
    return value is always non-negative.

    If /startSegment/ is zero and /numSegments/ is equal to the
    value of /path/'s PATH_COMMAND_COUNT_NV parameter and no error
    is generated, the value returned is identical to (equals) the value
    returned if GetPathParameterfvNV were used to query the value of
    /path/'s PATH_COMPUTED_LENGTH_NV parameter.

    The query

        boolean PointAlongPathNV(uint path,
                                 sizei startSegment, sizei numSegments,
                                 float distance,
                                 float *x, float *y,
                                 float *tangentX, float *tangentY);

    returns the point lying a given distance along a given portion of a
    path object specified by /path/ and the unit-length tangent vector
    at that point.  The boolean return value is TRUE if /distance/
    is within (inclusive) the arc length of the range of segments from
    /startSegment/ to /startSegment/+/numSegments/-1; otherwise FALSE
    is returned.

    The 2D point's (x,y) position is written to the values indicated
    by the /x/ and /y/ pointers respectively.  The tangent vector is
    written to the values indicated by the /tangentX/ and /tangentY/
    points.  However if /x/, /y/, /tangentX/, or /tangentY/ is a
    NULL pointer, no value is written to such NULL pointers.  Only the
    subpath consisting of the /numSegments/ path segments beginning with
    /startSegment/ (where the initial path segment has index 0) is used.
    PointAlongPathNV only considers this subpath.

    If /distance/ is less than or equal to zero, the starting point
    of the path is used (and the query returns FALSE).  If /distance/
    is greater than the path length (i.e., the value returned when the
    GetPathLengthNV query is called with the same /startSegment/ and
    /numSegments/ parameters), the final point along the subpath is used
    (and the query returns FALSE).

    The error INVALID_OPERATION is generated if /path/ does not exist.

    The error INVALID_VALUE is generated if /startSegment/
    or /numSegments/ are negative.  The error INVALID_VALUE is
    generated if /startSegment/ is greater than the index of /path/'s
    final path segment. The error INVALID_VALUE is generated if
    /startSegment/+/numSegments/-1 is less than zero or greater than
    the index of /path/'s final path segment.

    Because it is not possible in general to compute exact distances along
    a path, an implementation is not required to use exact computation
    even for segments where such computations are possible.

    Implementations are not required to compute distances exactly, as
    long as the satisfy the constraint that as /distance/ increases
    monotonically the returned point and tangent move forward
    monotonically along the path.

    Implementations should use the same distance-along-a-path algorithm
    for PointAlongPathNV as is used for dashing a stroked path.  (The dash
    count and dashing array state of the path object is irrelevant to
    the results of this query.)

    Where an implementation is able to determine that the point being
    queried lies exactly at a discontinuity or cusp, the incoming point
    and tangent should be returned.

    6.X.5. Path Color and Texture Coordinate Generation Queries

    The queries

        void GetPathColorGenivNV(enum color, enum pname, int *value);
        void GetPathColorGenfvNV(enum color, enum pname, float *value);

    return path color generation state.  /color/ must be one of
    PRIMARY_COLOR, PRIMARY_COLOR_NV, SECONDARY_COLOR_NV to return the
    requested color generation state for the primary, primary, and
    secondary color respectively.  /pname/ must be either PATH_GEN_MODE_NV,
    PATH_GEN_COLOR_FORMAT_NV, or PATH_GEN_COEFF_NV.

    If /pname/ is PATH_GEN_MODE_NV, the scalar value of the respective
    color's path generation mode is written to the value referenced by
    the /value/ pointer.

    If /pname/ is PATH_GEN_COLOR_FORMAT_NV, the scalar value of the
    respective color's path generation color format is written to the
    value reference by the /value/ pointer.

    If /pname/ is PATH_GEN_COEFF_NV, 16 coefficients for the respective
    color's path generation are written to the array referenced by the
    /value/ pointer.  Assuming no error is generated, 16 coefficients
    are written no matter what the path color generation mode is though
    coefficients not accessed by the indicated path color generation
    mode are returned as zero.

    The queries

        void GetPathTexGenivNV(enum texCoordSet, enum pname, int *value);
        void GetPathTexGenfvNV(enum texCoordSet, enum pname, float *value);

    return path texture coordinate set generation state.  /texCoordSet/
    indicates the texture coordinate set being queried and must be
    one of TEXTURE0 through TEXTUREn where /n/ is one less than the
    implementation-dependent value of MAX_TEXTURE_COORDS; otherwise
    INVALID_ENUM is generated.  /pname/ must be either PATH_GEN_MODE_NV
    PATH_GEN_COMPONENTS_NV, or PATH_GEN_COEFF_NV.

    If /pname/ is PATH_GEN_MODE_NV, the scalar value of the respective
    texture coordinate set's path generation mode is written to the
    value referenced by the /value/ pointer.

    If /pname/ is PATH_GEN_COMPONENTS_NV, the scalar value of the
    respective texture coordinate set's path generation number of
    components is written to the value reference by the /value/ pointer.

    If /pname/ is PATH_GEN_COEFF_NV, 16 coefficients for the respective
    texture coordinate set's path generation are written to the array
    referenced by the /value/ pointer.  Assuming no error is generated, 16
    coefficients are written no matter what the path texture generation
    mode is though coefficients not accessed by the indicated path
    texture generation mode are returned as zero.

Additions to the AGL/GLX/WGL Specifications

    Path objects are shared between AGL/GLX/WGL rendering contexts if
    and only if the rendering contexts share display lists.  No change
    is made to the AGL/GLX/WGL API.

    Changes to path objects shared between multiple rendering contexts
    will be serialized (i.e., the changes, queries, deletions, and
    stencil/cover operations will occur in a specific order).

Additions to the OpenGL Shading Language

    None

GLX Protocol

    XXX

Errors

    XXX

Dependencies on ARB_program_interface_query.

    When ARB_program_interface_query is not supported, all references to
    FRAGMENT_INPUT_NV and ProgramPathFragmentInputGenNV should be ignored.

Dependencies on Core Profile and OpenGL ES

    When NV_path_rendering is advertised, the following functionality
    must be supported...

    References to the following commands should be ignored:

        PathColorGenNV
        PathTexGenNV
        PathFogGenNV
        GetPathColorGenivNV
        GetPathColorGenfvNV
        GetPathTexGenivNV
        GetPathTexGenfvNV

    including the state set and queried by these commands.

    References to the following tokens should be ignored:

        PATH_FOG_GEN_MODE_NV
        PRIMARY_COLOR
        PRIMARY_COLOR_NV
        SECONDARY_COLOR_NV
        PATH_GEN_COLOR_FORMAT_NV

    References to the following GLSL built-in variables should be ignored:

        gl_TexCoord
        gl_MaxTextureCoords
        gl_Color
        gl_FrontColor
        gl_BackColor
        gl_SecondaryColor
        gl_FrontSecondaryColor
        gl_BackSecondaryColor
        gl_FogFragCoord

    The following types are defined as alias to the GL tokens:

        2_BYTES_NV                                      0x1407 // from GL compat
        3_BYTES_NV                                      0x1408 // from GL compat
        4_BYTES_NV                                      0x1409 // from GL compat
        EYE_LINEAR_NV                                   0x2400 // from GL compat
        OBJECT_LINEAR_NV                                0x2401 // from GL compat
        CONSTANT_NV                                     0x8576 // from GL compat

    The following entry points (specified by the EXT_direct_state_access
    extension) MUST be supported:

        void MatrixLoadfEXT(enum matrixMode, const float *m);
        void MatrixLoaddEXT(enum matrixMode, const double *m);

        void MatrixMultfEXT(enum matrixMode, const float *m);
        void MatrixMultdEXT(enum matrixMode, const double *m);

        void MatrixLoadTransposefEXT(enum matrixMode, const float *m);
        void MatrixLoadTransposedEXT(enum matrixMode, const float *m);

        void MatrixMultTransposefEXT(enum matrixMode, const float *m);
        void MatrixMultTransposedEXT(enum matrixMode, const float *m);

        void MatrixLoadIdentityEXT(enum matrixMode);

        void MatrixRotatefEXT(enum matrixMode, float angle,
                              float x, float y, float z);
        void MatrixRotatedEXT(enum matrixMode, double angle,
                              double x, double y, double z);

        void MatrixScalefEXT(enum matrixMode,
                             float x, float y, float z);
        void MatrixScaledEXT(enum matrixMode,
                             double x, double y, double z);

        void MatrixTranslatefEXT(enum matrixMode,
                                 float x, float y, float z);
        void MatrixTranslatedEXT(enum matrixMode,
                                 double x, double y, double z);

        void MatrixOrthoEXT(enum matrixMode, double l, double r,
                            double b, double t, double n, double f);
        void MatrixFrustumEXT(enum matrixMode, double l, double r,
                              double b, double t, double n, double f);

        void MatrixPushEXT(enum matrixMode);
        void MatrixPopEXT(enum matrixMode);

    These commands must support the PATH_PROJECTION_NV and PATH_MODELVIEW_NV
    tokens for matrixMode.  The associated modelview and projection matrix
    state, including matrix stacks, MUST be supported.  These token values
    for matrices are supported:

        PATH_PROJECTION_NV                                    0x1701
        PATH_MODELVIEW_NV                                     0x1700

        PATH_MODELVIEW_STACK_DEPTH_NV                         0x0BA3
        PATH_MODELVIEW_MATRIX_NV                              0x0BA6
        PATH_MAX_MODELVIEW_STACK_DEPTH_NV                     0x0D36
        PATH_TRANSPOSE_MODELVIEW_MATRIX_NV                    0x84E3
        PATH_PROJECTION_STACK_DEPTH_NV                        0x0BA4
        PATH_PROJECTION_MATRIX_NV                             0x0BA7
        PATH_MAX_PROJECTION_STACK_DEPTH_NV                    0x0D38
        PATH_TRANSPOSE_PROJECTION_MATRIX_NV                   0x84E4

    The last 8 tokens are supported by GetFloatv, GetIntegerv, GetDoublev
    to query associated path modelview and projection state.

    The values of sc, tc, rc, and qc discussed in section 5.X.2.2 "Path
    Covering" are always zero in a Core profile context as these values
    involve deprecated state.

New State

 -- NEW table 6.X, "Path (state per context)" following Table 6.33, "Renderbuffer"

    Get Value                            Type     Get Command          Initial Value   Description               Section       Attribute
    -----------------------------------  -------  -------------------  --------------  ------------------------  ------------  --------------
    PATH_GEN_MODE_NV                     2xZ4     GetPathColorGenivNV  NONE            path's color              6.X.5         lighting
                                                                                       generation mode
    PATH_GEN_COLOR_FORMAT_NV             2xZ6     GetPathColorGenivNV  NONE            path's color              6.X.5         lighting
                                                                                       generation color format
    PATH_GEN_COEFF_NV                    2x16xR   GetPathColorGenfvNV  all 0's         path's color gen mode     6.X.5         lighting
                                                                                       generation coefficients
    PATH_GEN_MODE_NV                     nxZ4     GetPathTexGenivNV    NONE            path's texture            6.X.5         texture
                                                                                       generation mode
    PATH_GEN_COMPONENTS_NV               nxZ5     GetPathTexGenivNV    0               path's texture            6.X.5         texture
                                                                                       generation number of
                                                                                       components
    PATH_GEN_COEFF_NV                    nx16xR   GetPathTexGenfvNV    all 0's         path's texture            6.X.5         texture
                                                                                       generation coefficients
    PATH_FOG_GEN_MODE_NV                 Z2       GetIntegerv          FRAGMENT_DEPTH  path's fog generation     5.X.2.1       fog
                                                                                       mode
    PATH_ERROR_POSITION_NV               Z        GetIntegerv          -1              last path string          5.X.1.2       -
                                                                                       error position
    PATH_STENCIL_FUNC_NV                 Z8       GetIntegerv          ALWAYS          path stenciling function  5.X.2.1       stencil-buffer
    PATH_STENCIL_REF_NV                  Z+       GetIntegerv          0               path stenciling           5.X.2.1       stencil-buffer
                                                                                       reference value
    PATH_STENCIL_VALUE_MASK_NV           Z+       GetIntegerv          1's             path stencil read mask    5.X.2.1       stencil-buffer
    PATH_STENCIL_DEPTH_OFFSET_FACTOR_NV  R        GetFloatv            0               path stencil depth        5.X.2.1       polygon
                                                                                       offset factor
    PATH_STENCIL_DEPTH_OFFSET_UNITS_NV   R        GetFloatv            0               path stencil depth        5.X.2.1       polygon
                                                                                       units factor
    PATH_COVER_DEPTH_FUNC_NV             Z8       GetIntegerv          LESS            path covering depth       5.X.2.2       depth-buffer
                                                                                       function

    where n is the implementation-dependent number of texture coordinate
    sets (MAX_TEXTURE_COORDS).

 -- NEW table 6.Y, "Path (state per path object)" following Table 6.X

    Get Value                        Type     Get Command           Initial Value         Description                       Section       Attribute
    -------------------------------  -------  --------------------  --------------------  --------------------------------  ------------  ---------
    -                                nxZ8*    GetPathCommandsNV     -                     path's sequence of path commands  6.X.1         -
    -                                mxR      GetPathCoordsNV       -                     path's sequence of path           6.X.1         -
                                                                                          coordinates
    -                                cxR      GetPathDashArrayNV    -                     dash array contents               5.X.1.5       -
    PATH_COMMAND_COUNT_NV            Z+       GetPathParameterivNV  -                     path's count of path commands     6.X.1         -
    PATH_COORD_COUNT_NV              Z+       GetPathParameterivNV  -                     path's count of path coordinates  6.X.1         -
    PATH_COMPUTED_LENGTH_NV          R+       GetPathParameterfvNV  -                     GL's calculation of the path's    6.X.1         -
                                                                                          length
    PATH_STROKE_WIDTH_NV             R+       GetPathParameterfvNV  1.0                   stroke width                      5.X.1.5       -
    PATH_INITIAL_END_CAP_NV          Z4       GetPathParameterivNV  FLAT                  path's initial end cap style      5.X.1.5       -
    PATH_TERMINAL_END_CAP_NV         Z4       GetPathParameterivNV  FLAT                  path's terminal end cap style     5.X.1.5       -
    PATH_JOIN_STYLE_NV               Z4       GetPathParameterivNV  MITER_REVERT_NV       path's join style                 5.X.1.5       -
    PATH_MITER_LIMIT_NV              R+       GetPathParameterfvNV  4                     path's miter limit                5.X.1.5       -
    PATH_DASH_ARRAY_COUNT_NV         Z+       GetPathParameterivNV  0                     path's count of dashes in the     5.X.1.5       -
                                                                                          path's dash array                 5.X.1.5       -
    PATH_DASH_OFFSET_NV              R        GetPathParameterfvNV  0.0                   path's dash offset                5.X.1.5       -
    PATH_DASH_OFFSET_RESET_NV        Z2       GetPathParameterivNV  MOVE_TO_CONTINUES_NV  path's dash offset reset          5.X.1.5       -
    PATH_CLIENT_LENGTH_NV            R+       GetPathParameterfvNV  0.0                   the client-supplied calculation   5.X.1.5       -
                                                                                          of the path's length
    PATH_INITIAL_DASH_CAP_NV         Z4       GetPathParameterivNV  FLAT                  path's initial dash cap style     5.X.1.5       -
    PATH_TERMINAL_DASH_CAP_NV        Z4       GetPathParameterivNV  FLAT                  path's terminal dash cap style    5.X.1.5       -
    PATH_FILL_MODE_NV                Z3       GetPathParameterivNV  COUNT_UP_NV           path's default fill mode          5.X.1.5       -
    PATH_FILL_MASK_NV                Z+       GetPathParameterivNV  all 1's               path's default fill mask          5.X.1.5       -
    PATH_FILL_COVER_MODE_NV          Z4       GetPathParameterivNV  CONVEX_HULL_NV        path's default fill cover mode    5.X.1.5       -
    PATH_STROKE_COVER_MODE_NV        Z4       GetPathParameterivNV  CONVEX_HULL_NV        path's default stroke cover mode  5.X.1.5       -
    PATH_STROKE_MASK_NV              Z+       GetPathParameterivNV  all 1's               path's default stroke mask        5.X.1.5       -
    PATH_STROKE_BOUND_NV             R[0,1]   GetPathParameterfvNV  0.2 (20%)             path's stroke approximation       5.X.1.5       -
                                                                                          bound
    PATH_OBJECT_BOUNDING_BOX_NV      R4       GetPathParameterfvNV  -                     path's outline bounding box       6.X.1         -
    PATH_FILL_BOUNDING_BOX_NV        R4       GetPathParameterfvNV  -                     path's fill bounding box          6.X.1         -
    PATH_STROKE_BOUNDING_BOX_NV      R4       GetPathParameterfvNV  -                     path's stroke bounding box        6.X.1         -

    where n is the number of commands in a path object, m is the number
    of coordinates in a path object, and c is the dash array count of
    a path object.

 -- NEW table 6.Z, "Path Glyph Metrics (state per path object)" following Table 6.Z

    Get Value                                Type     Get Command            Initial Value  Description                  Section       Attribute
    ---------------------------------------  -------  ---------------------  -------------  ---------------------------  ------------  ---------
    GLYPH_WIDTH_BIT_NV                       R        GetPathMetricsNV  see 5.X.1.3    path's glyph width                6.X.3         -
    GLYPH_HEIGHT_BIT_NV                      R        GetPathMetricsNV  see 5.X.1.3    path's glyph height               6.X.3         -
    GLYPH_HORIZONTAL_BEARING_X_BIT_NV        R        GetPathMetricsNV  see 5.X.1.3    path's glyph left side bearing    6.X.3         -
                                                                                       for horizontal layout
    GLYPH_HORIZONTAL_BEARING_Y_BIT_NV        R        GetPathMetricsNV  see 5.X.1.3    path's glyph top side bearing     6.X.3         -
                                                                                       for horizontal layout
    GLYPH_HORIZONTAL_BEARING_ADVANCE_BIT_NV  R        GetPathMetricsNV  see 5.X.1.3    path's glyph advance width        6.X.3         -
                                                                                       for horizontal layout
    GLYPH_VERTICAL_BEARING_X_BIT_NV          R        GetPathMetricsNV  see 5.X.1.3    path's glyph left side bearing    6.X.3         -
                                                                                       for vertical layout
    GLYPH_VERTICAL_BEARING_Y_BIT_NV          R        GetPathMetricsNV  see 5.X.1.3    path's glyph top side bearing     6.X.3         -
                                                                                       for vertical layout
    GLYPH_VERTICAL_BEARING_ADVANCE_BIT_NV    R        GetPathMetricsNV  see 5.X.1.3    path's glyph advance width        6.X.3         -
                                                                                       for vertical layout
    GLYPH_HAS_KERNING_BIT_NV                 B        GetPathMetricsNV  see 5.X.1.3    whether or not glyph has kerning  6.X.3         -
                                                                                       table

 -- NEW table 6.W, "Path Font Metrics (state per path object though identical for glyphs from the same font face)" following Table 6.Z

    Get Value                        Type     Get Command       Initial Value  Description                              Section       Attribute
    -------------------------------  -------  ----------------  -------------  ---------------------------------------  ------------  ---------
    FONT_X_MIN_BOUNDS_BIT_NV         R        GetPathMetricsNV  see 5.X.1.3    path's horizontal minimum (left-most)    6.X.3         -
                                                                               of the font bounding box
    FONT_Y_MIN_BOUNDS_BIT_NV         R        GetPathMetricsNV  see 5.X.1.3    path's vertical minimum (bottom-most)    6.X.3         -
                                                                               of the font bounding box
    FONT_X_MAX_BOUNDS_BIT_NV         R        GetPathMetricsNV  see 5.X.1.3    path's horizontal maximum (right-most)   6.X.3         -
                                                                               of the font bounding box
    FONT_Y_MAX_BOUNDS_BIT_NV         R        GetPathMetricsNV  see 5.X.1.3    path's vertical maximum (top-most)       6.X.3         -
                                                                               of the font bounding box
    FONT_UNITS_PER_EM_BIT_NV         R        GetPathMetricsNV  see 5.X.1.3    path's number of units in path space     6.X.3         -
                                                                               (font units) per Em square for font
                                                                               face
    FONT_ASCENDER_BIT_NV             R        GetPathMetricsNV  see 5.X.1.3    path's typographical ascender (in font   6.X.3         -
                                                                               units) of the font face
    FONT_DESCENDER_BIT_NV            R        GetPathMetricsNV  see 5.X.1.3    path's typographical descender (in font  6.X.3         -
                                                                               units) of the font face
    FONT_HEIGHT_BIT_NV               R+       GetPathMetricsNV  see 5.X.1.3    path's font face vertical distance       6.X.3         -
                                                                               between two consecutive baselines
                                                                               (in font units)
    FONT_MAX_ADVANCE_WIDTH_BIT_NV    R        GetPathMetricsNV  see 5.X.1.3    path's maximal advance width (in font    6.X.3         -
                                                                               units) for all glyphs in font face
    FONT_MAX_ADVANCE_HEIGHT_BIT_NV   R        GetPathMetricsNV  see 5.X.1.3    path's maximal advance height (in font   6.X.3         -
                                                                               units) for all glyphs in font face
                                                                               for vertical layout
    FONT_UNDERLINE_POSITION_BIT_NV   R        GetPathMetricsNV  see 5.X.1.3    path's position (in font units) of the   6.X.3         -
                                                                               center of underline line for font face
    FONT_UNDERLINE_THICKNESS_BIT_NV  R        GetPathMetricsNV  see 5.X.1.3    thickness (in font units) of the         6.X.3         -
                                                                               underline for font face
    FONT_HAS_KERNING_BIT_NV          B        GetPathMetricsNV  see 5.X.1.3    whether or not glyph has kerning         6.X.3         -
                                                                               table
    FONT_NUM_GLYPH_INDICES_BIT_NV    Z+       GetPathMetricsNV  see 5.X.1.3    number of glyph indices in font face     6.X.3         -

    Increment "n" in the Type field of the "Program Interface State"
    table by 1 to correspond to the FRAGMENT_INPUT_NV program interface.

    Add the following rows to the table labeled "Program Object Resource
    State" (only fragment input resources support this state):

                                                        Initial
    Get Value               Type  Get Command           Value      Description                                  Sec.
    ----------------------  ----  --------------------  ---------  -------------------------------------------  -----
    PATH_GEN_MODE_NV         Z4    GetProgramResourceiv  NONE       Path fragment input generation mode
    PATH_GEN_COMPONENTS_NV   Z5    GetProgramResourceiv  0          Number of path fragment input components
    PATH_GEN_COEFF_NV        16*R  GetProgramResourceiv  all zeros  Path fragment input generation coefficients

New Implementation Dependent State

    None

NVIDIA Implementation Details

 -- API revision 1.0

    Released in NVIDIA Driver Release 275.33 (June 2011).

 -- API revision 1.1

    Follow-on release (circa September 2011) adds these path
    commands for ISO PDF 32000 support:

        GL_RESTART_PATH_NV
        GL_DUP_FIRST_CUBIC_CURVE_TO_NV
        GL_DUP_LAST_CUBIC_CURVE_TO_NV
        GL_RECT_NV

    These path commands are not operational (generate GL_INVALID_ENUM
    errors) if used in 275.xx or 280.xx drivers.

    Follow-on release (circa September 2011) adds these transformType
    parameters:

        GL_NONE
        GL_TRANSLATE_3D_NV
        GL_AFFINE_3D_NV
        GL_TRANSPOSE_AFFINE_3D_NV

    These transformType tokens are not operational (generate
    GL_INVALID_ENUM errors) if used in 275.xx or 280.xx drivers.

 -- API revision 1.2

    Follow-on release (circa December 2013) adding these commands:

        void glMatrixLoad3x2fNV(GLenum matrixMode, const GLfloat *m);
        void glMatrixLoad3x3fNV(GLenum matrixMode, const GLfloat *m);
        void glMatrixLoadTranspose3x3fNV(GLenum matrixMode, const GLfloat *m);

        void glMatrixMult3x2fNV(GLenum matrixMode, const GLfloat *m);
        void glMatrixMult3x3fNV(GLenum matrixMode, const GLfloat *m);
        void glMatrixMultTranspose3x3fNV(GLenum matrixMode, const GLfloat *m);

        void glStencilThenCoverFillPathNV(GLuint path, GLenum fillMode,
                                          GLuint mask, GLenum coverMode);
        void glStencilThenCoverStrokePathNV(GLuint path, GLint reference,
                                            GLuint mask, GLenum coverMode);
        void glStencilThenCoverFillPathInstancedNV(GLsizei numPaths,
                                                   GLenum pathNameType,
                                                   const void *paths,
                                                   GLuint pathBase,
                                                   GLenum fillMode, uint mask,
                                                   GLenum coverMode,
                                                   GLenum transformType,
                                                   const GLfloat *transformValues);
        void glStencilThenCoverStrokePathInstancedNV(GLsizei numPaths,
                                                     GLenum pathNameType,
                                                     const void *paths,
                                                     GLuint pathBase,
                                                     GLint reference, uint mask,
                                                     GLenum coverMode,
                                                     GLenum transformType,
                                                     const GLfloat *transformValues);
        enum glPathGlyphIndexRangeNV(GLenum fontTarget,
                                     const void *fontName,
                                     GLbitfield fontStyle,
                                     GLuint pathParameterTemplate,
                                     GLfloat emScale,
                                     GLuint baseAndCount[2]);

    If the window system's GetProcAddress mechanism for GL commands returns
    NULL for these function names, these API revision 1.2 features are
    not available.  Likewise the these tokens are not supported either.

        GL_ROUNDED_RECT_NV
        GL_RELATIVE_ROUNDED_RECT_NV
        GL_ROUNDED_RECT2_NV
        GL_RELATIVE_ROUNDED_RECT2_NV
        GL_ROUNDED_RECT4_NV
        GL_RELATIVE_ROUNDED_RECT4_NV
        GL_ROUNDED_RECT8_NV
        GL_RELATIVE_ROUNDED_RECT8_NV
        GL_RELATIVE_RECT_NV

    These tokens may be returned by glPathGlyphIndexRangeNV:

        GL_FONT_GLYPHS_AVAILABLE_NV
        GL_FONT_TARGET_UNAVAILABLE_NV
        GL_FONT_UNAVAILABLE_NV
        GL_FONT_UNINTELLIGIBLE_NV

 -- API revision 1.3

    Follow-on release (circa May 2014, first appearing in the 337.88
    drivers) adding these commands:

    These new path commands:

        GL_CONIC_CURVE_TO_NV
        GL_RELATIVE_CONIC_CURVE_TO_NV
        GL_RELATIVE_RECT_NV

    New path glyph metric query:

        GL_FONT_NUM_GLYPH_INDICES_BIT_NV

    New return values from glyph index path specification commands:

        GL_FONT_UNINTELLIGIBLE_NV
        GL_STANDARD_FONT_FORMAT_NV

    New programInferface token:

        FRAGMENT_INPUT_NV

    New (aliased) matrix tokens for ES support:

        PATH_PROJECTION_NV
        PATH_MODELVIEW_NV

        PATH_MODELVIEW_STACK_DEPTH_NV
        PATH_MODELVIEW_MATRIX_NV
        PATH_MAX_MODELVIEW_STACK_DEPTH_NV
        PATH_TRANSPOSE_MODELVIEW_MATRIX_NV
        PATH_PROJECTION_STACK_DEPTH_NV
        PATH_PROJECTION_MATRIX_NV
        PATH_MAX_PROJECTION_STACK_DEPTH_NV
        PATH_TRANSPOSE_PROJECTION_MATRIX_NV

    New glyph index path specification commands:

        enum glPathGlyphIndexArrayNV(GLuint firstPathName,
                                     GLenum fontTarget,
                                     const void *fontName,
                                     GLbitfield fontStyle,
                                     GLuint firstGlyphIndex,
                                     GLsizei numGlyphs,
                                     GLuint pathParameterTemplate,
                                     GLfloat emScale);
        enum glPathMemoryGlyphIndexArrayNV(GLuint firstPathName,
                                           GLenum fontTarget,
                                           GLsizeiptr fontSize,
                                           const void *fontData,
                                           GLsizei faceIndex,
                                           GLuint firstGlyphIndex,
                                           GLsizei numGlyphs,
                                           GLuint pathParameterTemplate,
                                           GLfloat emScale);

    GLSL-related commands:

        void glProgramPathFragmentInputGenNV(GLuint program,
                                             GLint location,
                                             GLenum genMode,
                                             GLint components,
                                             const GLfloat *coeffs);

        void glGetProgramResourcefvNV(GLuint program, GLenum programInterface,
                                      GLuint index, GLsizei propCount,
                                      const GLenum *props, GLsizei bufSize,
                                      GLsizei *length, GLfloat *params);

    If the window system's GetProcAddress mechanism for GL commands returns
    NULL for these function names, these API revision 1.3 features are
    not available.  Likewise the these tokens are not supported either.

 -- Performance improvements:

    Release 304.xx and on substantially improves the performance of
    path rendering stencil and cover operations on NVIDIA Fermi- and
    Kepler-based GPUs (GeForce 400 Series and on).

    Release 314.xx improves the performance of initially specifying or
    modifying a path object prior to stenciling or covering the paths.

 -- Bugs:

    Due to NVIDIA driver bug 1315267, path objects were not actually
    shared among contexts prior to Driver release 320.xx (July 2013).

Issues

    1.  What should this extension be called?

        RESOLVED:  NV_path_rendering

        The extension adds an entirely new rendering paradigm for filled
        and stroked paths, hence "path rendering".

        "path" alone was considered but deemed to vague.

    2.  Should this extension support specifying paths based on glyphs in fonts?

        RESOLVED:  Yes.

        There are several problems solved by including first-class path
        specification via glyphs in fonts.

        1)  Support for glyphs from fonts are an expected part of nearly
        all path rendering systems.  Not including glyph support will force
        all path rendering applications to build their own glyph system.

        2)  Fonts, particularly for Asian languages, can be large.
        By putting glyph specification from fonts directly into
        the extension, implementations will have the opportunity to
        cache commonly loaded font glyphs, including shared on-GPU
        representations.

        3)  Also because fonts have many glyphs, first-pass specification
        of a range of glyphs allows the GL implementation to load glyphs
        sparsely in response to use.  It isn't appropriate to burden
        applications with the burden of properly caching large sets of
        glyphs from fonts.  So while Unicode glyphs 0 through 2^21-1
        might be loaded for an entire Unicode font, the GL implementation
        could only actually load queried and used path objects.

        4)  Locating font files tends to be very system-specific.  To the
        extent OpenGL supports cross-platform rendering, minimizing
        system-specific aspects of rendering increases the cross-platform
        nature of OpenGL applications.

        5)  I still feel bad about glutBitmapCharacter and
        glutStrokeCharacter being so lame; I thought something better
        would take their place but nothing has.

    3.  Should this extension provide a one-shot (single-pass) way to fill
        and stroke path objects?

        RESOLVED:  No.

        The two-pass decoupling of path rendering into stenciling and
        covering operations has lots of advantages.

        Some of the advantages are:

        1)  The cover step has complete control over how the fragment
            shader is configured.  GLSL, assembly, or fixed-function
            (glTexEnv, glFog, etc.) fragment shading can all be used
            without conflating the path's coverage determination with
            the shading.

        1a) If shading resources are used to implement the path coverage
            determination such as interpolants or textures, these
            resources aren't over-subscribed as could occur in a
            one-step API.

        2)  GPUs can accelerate stencil-only rendering during the stencil
            step in ways that are rasterization and bandwidth efficient.

        3)  Path rendering standards all allow a rendered path to be clipped
            by another arbitrary (clipping) path.  This can even be
            done recursively sometimes.  When the stencil coverage
            determination is a separate step from the shading, such
            clipping operations are easy to accomplish as simply multiple
            stenciled stencil steps.  Otherwise clipping a path to another
            path is a complex intersection and re-tessellation task.

        4)  A two-step stencil, then cover" approach makes it
            straightforward to guarantee that pixels and samples are
            only visited once per path rendering as path rendering
            standards require.

        5)  Unconventional but efficient algorithms such as reverse
            Painter's algorithm are straightforward to implement when
            stencil and covering steps are decoupled.  In this case,
            the stencil buffer simply never allows a pixel to be shaded
            a second time once covered.

        6)  Dilations and erosions can be performed with the two-step
            approach.  You can fill a shape and then stroke the shape
            to dilate the shape.  Then cover with both fill and stroking.

        7)  Novel stroking effects such as pin-striping are easy to
            accomplish.  You can stroke a path with a stroke width of 7.5
            to write stencil to 1 and then stroke the same curve with
            a stroke width of 3.1 to write a stencil of 0.  Then cover
            stroked path with a stroke width of 7.5 and accomplish the
            pen-striping.

    4.  What string formats should be supported for paths?

        UNRESOLVED:  Definitely SVG and PostScript.  Perhaps Silverlight?

        Silverlight's path syntax is very similar to SVG but allows
        infinity values and the specification of the fill mode.

        Adobe's Type 2 charstring format, part of Adobe's Compact
        Font Format (CFF) standard, provides yet another encoding
        of a path outline.  This is a binary, rather than textual,
        format that exists within OpenType and Type 2 font formats.
        It includes glyph hinting information.  The utility of accepting
        the Type 2 charstring format is not sufficient for inclusion for
        a number of reasons.  Content creation tools don't target this
        format for arbitrary path descriptions.  This extension already
        provides commands (glPathGlyphsNV and glPathGlyphRangeNV) for
        specifying path objects from font glyphs.  And the font hinting
        information the format provides would just be ignored.

    5.  Should there be a query to return a path as a string?

        RESOLVED: No, returning dynamic strings of variable length is too hard.

        Unlike parsing which is straightforward but slightly difficult,
        building a string from the results of glGetPathCommandsNV and
        glGetPathCoordsNV is not hard.  But there's no one "right"
        way to build a string for a given path.

        There are  various string encodings of varying compactness and
        readability.  How much precision is really required for converting
        a floating-point coordinate to a string representation varies?

    6.  Should path rendering allow per-vertex specification of attributes?

        RESOLVED:  No.

        There is no allowance for the specification of interpolants
        (colors, texture coordinates, etc.) specified per control point.
        Assigning per-vertex values to control points doesn't really
        make sense in the context of path rendering.  Instead a mechanism
        for generating color, texture coordinate, and fog generation as
        a linear function of object space.

        The glPathTexGenNV, glPathColorGenNV, and glPathFogGenNV commands
        provide this mechanism.

        glPathColorGenNV, glPathTexGenNV, and glPathFogGenNV
        provide a way to specify coefficients for plane equations based
        on the object-space or eye-space (x,y,1) position of a fragment
        generating by covering a filled or stroked path.

        For those used to conventional 3D graphics where geometry is
        defined by meshes of triangles, not having per-vertex attributes
        sounds really strange.  But this is the natural situation for
        path rendering.  Paths do not really have vertices but rather
        control points.

    7.  Should path rendering use the existing texture coordinate
        generation state (glTexGen)?

        RESOLVED:  No, this extension should have its own path-specific
        texture coordinate generation state controlled by glPathColorGenNV
        and glPathTexGenNV.

        The existing texture coordinate generation state has modes such
        as sphere, normal, and reflection mapping that make no sense for
        path rendering (since there are no per-vertex normals).

        Also it is very desirable to keep the per-vertex attribute
        computations (normal transformation, lighting, texture coordinate
        generation) completely separate from the varying computations
        for path rendering.  This means the vertex processing program
        needed for path rendering isn't changed by state updates intended
        to control geometric and image primitives.

    8.  How does path rendering work if all the fixed-function state,
        particularly the modelview and projection matrices and named vertex
        attributes (primary color, etc.) have been deprecated?

        RESOLVED:  ARB_compatibility (for OpenGL 3.1) or the Compatibility
        profile (for OpenGL 3.2 and up) is useful and supports the
        glPathColorGenNV, glPathTexGenNV, and glPathFogGenNV commands
        to interact properly with fixed-function OpenGL.

        NV_path_rendering assumes that the modelview and projection
        matrices combine to transform the path into clip space.
        Without these matrices, there's no way to get the path
        transformed.  Therefore these matrices are introduced through
        EXT_direct_state_access commands when only the Core profile is
        supported.

        Without ARB_compatibility or the Compatibility profile, there is
        no way for GLSL to access built-in varyings as these have been
        deprecated.  This means generated or passed-through colors and
        texture coordinate sets are inaccessible.  There's also no longer
        a way to compile a fragment shader that doesn't have a vertex
        shader.  The ARB_separate_shader_objects extension (core in OpenGL
        4.1) now allows a fragment shader to be specified in a program
        object with a vertex shader.  The ARB_program_interface_query
        extension (core in OpenGL 4.3) allows queries to specific program
        object resources.  The glProgramPathFragmentInputGenNV provides
        a means, in combination with the ARB_separate_shader_objects and
        ARB_program_interface_query extensions, to configure fragment
        input generation for GLSL fragment shaders use during the "cover"
        step of path rendering without reference to fixed-function
        mechanisms.

        See the "Dependencies on Core Profile and OpenGL ES" section
        and issue 133 for more details about the Core profile support.

    9.  How does a GLSL fragment shader processing fragment generated by
        covering path access fragment varyings?

        RESOLVED:  The obvious way is to used the gl_TexCoord[], gl_Color,
        and gl_SecondaryColor built-in varyings for texture coordinate
        sets, primary color, and secondary color respectively.

        Any user-specified varyings will be undefined since there is no
        upstream geometry or vertex shader to write them.

    10. How should the command token values be assigned?

        RESOLVED:  Consistent with OpenVG's enumeration values but ALSO using
        the SVG command characters too.

        The two token addition (missing from OpenVG's supported commands)
        are relative and absolute 7-component partial elliptical arc
        tokens (GL_ARC_TO_NV and GL_RELATIVE_ARC_TO_NV) that include the
        large/small and sweep flags as coordinates.  These corresponds
        to the SVG 'arcto' commands.

        Using character codes, in addition to tokens, allows simpler
        path descriptions coded with C character arrays (strings) such
        as "MLLCz" instead of the equivalent verbose aggregate array
        initializer { GL_MOVE_TO_NV, GL_LINE_TO_NV, GL_LINE_TO_NV,
        GL_CUBIC_CURVE_TO_NV, GL_CLOSE_PATH_NV }.

        Also note there are NO character codes for the eight 5-component
        partial elliptical arc commands because these commands lack
        exact analogues in the SVG path command syntax.

        There are also commands corresponding to PostScript's circular arc
        commands (arc, arcn, and arct), also without character aliases.

        Unfortunately the path command token values do NOT match the
        SVGPathSeg interface path segment type values because these
        values overlap with the OpenVG enumeration values.

        Excepting the printable ASCII character command tokens, absolute
        command token values should always be even, while relative
        command token values should always be odd.

    11. Why are the glyph metric bits in the order they are specified?

        RESOLVED:  The glyph metric order matches the FreeType 2 library's
        FT_Glyph_Metrics structure for the per-glyph metrics. The
        per-font metric order matches the FreeType 2 library's FT_FaceRec
        structure.

        Kerning information for a font face can be queried with the
        separate query glGetPathSpacingNV because the kerning displacement
        is not per-glyph, but rather dependent on a sequence of two
        glyphs.

    12. What glyph metric information is beyond the scope of this extension?

        RESOLVED:  Metrics for vertical kerning, bi-directional layout,
        ligatures, etc.  are beyond the scope of this extension.

        Kerning information for horizontal layout is available.

        The scope of the metrics provided by this extension are sufficient
        for basic kerned and non-kerned horizontal and non-kerned vertical
        text layout.

        Applications that want more sophisticated metric information
        should either query the metrics from the corresponding system
        font directly or load the glyph outline data entirely from the
        application.

    13. What glyph outline information is beyond the scope of this
        extension?

        RESOLVED:  For now, normal (indicated by GL_NONE), italic,
        bold, and bold/italic font faces are supported.  Other styles
        (small caps, etc.) may be added with future extensions to this
        extension, but the four supported font styles are sufficient.

        This is consistent with FreeType 2's support for  the
        FT_STYLE_FLAG_ITALIC and FT_STYLE_FLAG_BOLD flags.

    14. Should horizontal kerning information always be available?

        RESOLVED:  Yes, if the font provides this kerning information.

    15. Why is the horizontal kerning information for a pair of path
        objects returned as a 2D (x,y) displacement?

        RESOLVED:  TrueType fonts always return kerning information
        as a sequence of horizontal displacements in x, but not y (the
        displacement is assumed to be zero in y).  However PostScript
        fonts can support a 2D displacement.

        This matches the behavior of FreeType 2's FT_Get_Kerning function.

        Note that the returned (x,y) float pairs are NOT immediately
        suitable to be used as values for the /transformValues/
        array parameter to StencilFillPathInstancedNV,
        StencilStrokePathInstancedNV, CoverFillPathInstancedNV, or
        CoverStrokePathInstancedNV with a /transformType/ parameter
        of TRANSLATE_2D_NV.  The application would be responsible for
        accumulating the various translates to provide proper horizontal
        layout.  When all the y values are zero (as will often be the
        case), GL_TRANSLATE_1D_NV could be used instead.

    16. Should the path name zero be treated specially?

        RESOLVED:  No.  There's no need for specially handling the zero
        name for a path object.

    17. What tokens for color should glPathColorGenNV accept?

        RESOLVED:  GL_PRIMARY_COLOR (from OpenGL 1.3), GL_PRIMARY_COLOR_NV
        (from NV_register_combiners), and GL_SECONDARY_COLOR_NV (from
        NV_register_combiners).

        GL_PRIMARY_COLOR and GL_PRIMARY_COLOR_NV have different token
        values; to avoid an API pitfall, both are accepted.

        (There is no core GL_SECONDARY_COLOR token.)

    18. Should two-sided color be supported for path rendering?

        RESOLVED:  No.  No path rendering standards support this concept.

        Two-sided lighting could be simulated with two passes and face
        culling.

    19. How do PostScript's user path operators correspond to
        NV_path_rendering's path command tokens?

        RESOLVED:

            PostScript path
            operator         Path command token
            ---------------  -----------------------------
            arc              GL_CIRCULAR_CCW_ARC_TO_NV
            arcn             GL_CIRCULAR_CW_ARC_TO_NV
            arcto            GL_CIRCULAR_TANGENT_ARC_TO_NV
            closepath        GL_CLOSE_PATH_NV
            curveto          GL_CUBIC_CURVE_TO_NV
            lineto           GL_LINE_TO_NV
            moveto           GL_MOVE_TO_NV
            rcurveto         GL_RELATIVE_CUBIC_CURVE_TO_NV
            rlineto          GL_RELATIVE_LINE_TO_NV
            rmoveto          GL_RELATIVE_MOVE_TO_NV
            setbbox          /ignored/
            ucache           /ignored/

        The setbbox (set bounding box) operator "establishes a bounding
        box for the current path, within which the coordinates of all
        subsequent path construction operators must fall."  There is
        no such requirement in this extension so this bounding box
        information is parsed but ignored.

        The ucache (user cache) operator "notifies the PostScript
        interpreter that the enclosing user path is to be retained in
        the cache if the path is not already there."  This notion that
        paths are expensive and so must be cached is not particularly
        applicable to this extension because all path object are in
        some sense cached.  Therefore the ucache operator is parsed
        but ignored.

    20. How do OpenVG 1.1's commands (as enumerated by the VGPathCommand
        and VGPathCommand enumerations) correspond to NV_path_rendering's
        path command tokens?

        RESOLVED:

            OpenVG path
            segment command    Path command token
            -----------------  ----------------------------------------
            VG_CLOSE_PATH      GL_CLOSE_PATH_NV
            VG_CUBIC_TO        GL_CUBIC_CURVE_TO_NV
            VG_CUBIC_TO_ABS    "
            VG_CUBIC_TO_REL    GL_RELATIVE_CUBIC_CURVE_TO_NV
            VG_HLINE_TO        GL_HORIZONTAL_LINE_TO_NV
            VG_HLINE_TO_ABS    "
            VG_HLINE_TO_REL    GL_RELATIVE_HORIZONTAL_LINE_TO_NV
            VG_LCCWARC_TO      GL_LARGE_CCW_ARC_TO_NV
            VG_LCCWARC_TO_ABS  "
            VG_LCCWARC_TO_REL  GL_RELATIVE_LARGE_CCW_ARC_TO_NV
            VG_LCWARC_TO       GL_LARGE_CW_ARC_TO_NV
            VG_LCWARC_TO_ABS   "
            VG_LCWARC_TO_REL   GL_RELATIVE_LARGE_CW_ARC_TO_NV
            VG_LINE_TO         GL_LINE_TO_NV
            VG_LINE_TO_ABS     "
            VG_LINE_TO_REL     GL_RELATIVE_LINE_TO_NV
            VG_MOVE_TO         GL_MOVE_TO_NV
            VG_MOVE_TO_ABS     "
            VG_MOVE_TO_REL     GL_RELATIVE_MOVE_TO_NV
            VG_QUAD_TO         GL_QUADRATIC_CURVE_TO_NV
            VG_QUAD_TO_ABS     "
            VG_QUAD_TO_REL     GL_RELATIVE_QUADRATIC_CURVE_TO_NV
            VG_SCCWARC_TO      GL_SMALL_CCW_ARC_TO_NV
            VG_SCCWARC_TO_ABS  "
            VG_SCCWARC_TO_REL  GL_RELATIVE_SMALL_CCW_ARC_TO_NV
            VG_SCUBIC_TO       GL_SMOOTH_CUBIC_TO_NV
            VG_SCUBIC_TO_ABS   "
            VG_SCUBIC_TO_REL   GL_RELATIVE_SMOOTH_CUBIC_TO_NV
            VG_SCWARC_TO       GL_SMALL_CW_ARC_TO_NV
            VG_SCWARC_TO_ABS   "
            VG_SCWARC_TO_REL   GL_RELATIVE_SMALL_CW_ARC_TO_NV
            VG_SQUAD_TO        GL_SMOOTH_QUADRATIC_CURVE_TO_NV
            VG_SQUAD_TO_ABS    "
            VG_SQUAD_TO_REL    GL_RELATIVE_SMOOTH_QUADRATIC_CURVE_TO_NV
            VG_VLINE_TO        GL_VERTICAL_LINE_TO_NV
            VG_VLINE_TO_ABS    "
            VG_VLINE_TO_REL    GL_RELATIVE_VERTICAL_LINE_TO_NV

    21. What should the initial GL_PATH_FILL_MODE_NV state be?

        RESOLVED:  GL_PATH_FILL_MODE_NV should initially be
        GL_COUNT_UP_NV.

        This is consistent with SVG default non-zero fill rule and the
        typical usage of PostScript.

        However this is the opposite of Silverlight's default fill rule
        which is even-odd.

    22. Should we support a GL_PATH_FORMAT_SL3_NV for Silverlight 3.0 be
        added?

        UNRESOLVED:  Silverlight 3.0's path markup syntax includes
        support for two extensions of the SVG 1.1 path grammar:  1)
        specification of a fill rule ("F0" is even-odd, "F1" is non-zero)
        at the beginning of the string; and 2) allowing "Infinity",
        "-Infinity" and "NaN" (all case-sensitive) as special values
        allowed instead of standard numerical values.

        Seems like a reasonable thing to support.

        The fill rule specification is straightforward.  This would
        simply allow the path object's GL_PATH_FILL_MODE_NV parameter to
        be specified as part of the path string specification.  The "F0"
        or "F1" would not be treated as an actual path command however.

        Note that Silverlight's default fill rule if none is specified is
        EvenOdd whereas NV_path_rendering's default GL_PATH_FILL_MODE_NV
        is GL_COUNT_UP_NV (essentially a non-zero rule).  So specifying a
        path with the Silverlight format would have a different initial
        value for GL_PATH_FILL_MODE_NV if "F0" or no initial F
        command is specified.

        What exactly would it mean to allow infinite and not-a-number
        values for the coordinate values of a path object?  Infinite is
        probably representable today by simply writing a number with a
        sufficiently large enough magnitude.  Allowing not-a-number is
        probably more

    23. Should there be some maximum specified limit for the number of
        command (and hence coordinates) in a path object?

        RESOLVED:  No.  The standards for path rendering do no generally
        have limits on path command lengths.

        For extreme cases, the OUT_OF_MEMORY error would be generated but
        that is considered an exceptional case due to memory exhaustion,
        not simply the specification of a huge path.

    24. Should there be some maximum specified limit for the dash array?

        RESOLVED:  No.  The standards for path rendering do not generally
        have a limit on the dash array length.

    25. How do the client-defined clip planes and the clip volume interact
        with path rendering?

        RESOLVED:  Stenciled and covered paths are affected by both the
        clip volume and client-defined clip planes.

        Clip planes affect the set of accessible pixels for stenciling
        and covering operations (see the "ACCESSIBLE SAMPLES WITH RESPECT
        TO A TRANSFORMED PATH" subsection of section 5.X.2.1).

    26. What should the end cap style be called that adds no additional
        cap region to a stroked path?

        RESOLSVED:  FLAT.

        FLAT is already an existing OpenGL token name.  Silverlight and
        the OpenXML Paper Specification (XPS) calls this end cap style
        "flat".  However PostScript, Flash, OpenVG, SVG, Quartz 2D,
        and Cairo Graphics all call this end cap style "butt".

        Using FLAT also avoids verbalizing the awkward phrase "butt
        stroking" (not that there's anything wrong with that).

    27. Should the PostScript grammar for user-defined paths be supported?

        RESOLVED:  Yes.  PostScript has commands (arc, arcn, arct) that
        do not correspond to precisely to SVG command.  This is
        particularly true of arct.

        Applications have been generating paths according to the syntax
        of Level 2 PostScript for a long time.

        Level 2 PostScript also has support for binary encoding that
        makes it significantly more compact and less expensive to parse
        than the SVG grammar.

        The binary encoding allows precise floating-point (and compact
        fixed-point) values to be specified.

    28. Should the PostScript grammar support big-endian, little-endian,
        and native numeric encodings?

        RESOLVED:  Yes, yes, yes.

    29. Should the PostScript grammar support encoded user paths?

        RESOLVED:  Yes.

    30. How should the PostScript grammar support strings?

        UNRESOLVED:  Hexadecimal encoded data, that is strings enclosed
        in < and >, are supported.

        Strings for ASCII base-85 encoded data, that is strings enclosed
        in <~ and ~>, are supported for the data-array and operator-string
        production

        Also the short-binary-string, be-long-binary-string, and
        le-long-binary-string productions allow very compact and precise
        encoding of operator strings through binary encoding.

        Strings for literal text, that is strings enclosed in ( and ),
        are NOT supported.

        The rationale for not supporting literal text is this format
        is awkward for encoding the operator-string production (though
        PostScript does technically allow it) and is not compact.

    31. Should the PostScript grammar support Binary Object Sequences?

        RESOLVED:  No.

        Binary Object Sequences are intended to support complex
        (potentially nested) data structures and are over-kill for
        user paths.

    32. Why are the binary tokens in the PS grammar assigned the values
        they are assigned?

        RESOLVED:  These values are from the "Binary Tokens" section of
        the PostScript Language Reference Manual.

    33. Why are the binary encodings for the path commands in the PS
        grammar assigned the specified values?

        RESOLVED:  These values match PostScript's system name table
        values.  These are documented in the "System Name Encodings"
        appendix of the PostScript Language Reference Manual.
        Specifically (in decimal):

            Index  Name
            -----  ---------
            22     closepath
            99     lineto
            107    moveto
            133    rlineto
            134    rmoveto
            143    setbbox
            43     curveto
            122    rcurveto
            5      arc
            6      arcn
            7      arct
            177    ucache

    34. Why do glGetPathCommandsNV, glGetPathCoordsNV, and
        glGetPathDashArrayNV have their own queries?  Could there not
        simply be a token for glGetPathParameteriv/glGetPathParameterfvNV
        to return this state?

        RESOLVED:  These queries for path commands, coordinates, and
        the path's dash array return a variable payload of data so are
        more like glGetTexImage than glGetIntegerv/glGetFloatv which
        return a static amount of data.

        APIs that return variable amounts of data are prone to buffer
        overflows.  It is somewhat more obvious these commands return
        a variable amount of data if they have their own API calls, than
        simply having certain token values to a multi-purpose glGet* call
        that mysteriously returns varying amounts of data for these token
        values while all the other tokens return static amounts of data.

        This resolution follows the existing precedent from
        core OpenGL where glGetColorTable is distinct from
        glGetColorTableParameter{fv,iv}.  Same with glGetConvolutionFilter
        and glGetHistogram relative to glGetConvolutionParameter{fv,iv}
        and glGetHistogramParameter{fv,iv}.

        (There is a poor precedent for having an OpenGL query return both
        static and varying amounts of data based on a pname parameter.
        glGetMap{dv,fv,iv} returns varying data when GL_COEFF is queried
        while GL_ORDER and GL_DOMAIN return n and 2*n values respectively
        where n is the dimensionality of the map target.  This isn't a
        good precedent and is obscure.)

    35. How should the GL_PATH_*_BOUNDING_BOX_NV path parameters be
        returned?

        RESOLVED:  In (x1,y1,x2,y2) order where (x1,y1) is the minimum
        bounds of the bounding box and (x2,y2) is the maximum bounds.

        This is contrary to the precedent of GL_SCISSOR_BOX query
        which returns the scissor as an (x,y,width,height) 4-tuple.
        While that makes sense for a scissor box, particularly given how
        the scissor is specified with glScissor, it is not a convenient
        way to specify a bounding box.

        The (x1,y1,x2,y2) format also makes the
        glCover{Fill|Stroke}PathInstancedNV pseudo-code work nicely
        with glRectf.  See the renderBoundingBox pseudo-code.

        The (x1,y1,x2,y2) format is also consistent with the way
        FreeType 2 provides per-font face bounds information through
        the GL_FONT_X_MIN_BOUNDS_BIT_NV, GL_FONT_Y_MIN_BOUNDS_BIT_NV,
        GL_FONT_X_MAX_BOUNDS_BIT_NV, and GL_FONT_Y_MAX_BOUNDS_BIT_NV
        metric queries.

    36. Why is font loading part of this extension?  Shouldn't OpenGL
        stick with just rendering and not involved itself with fonts?

        RESOLVED:  An explicit goal of this extension is to provide
        GPU-accelerated path rendering that INCLUDES excellent support
        for glyphs and their associated metrics.

        The fact is all the major existing standards for path rendering
        (PostScript, SVG, OpenVG, Java 2D, Quartz 2D, Flash) include
        first-class font and glyph support.

        Not including font and glyph support would be a glaring omission
        that would make this extension much less useful to simple OpenGL
        applications that don't want to incorporate large font libraries.

        Additionally font loading is notoriously platform dependent.
        This extension provides a simple platform-independent mechanism to
        rendezvous with standard font names.  However an implementation of
        this extension can make use of whatever platform-specific font
        services the platform provides (such as through DirectWrite,
        etc.).

        Fonts, particularly for Asian languages or designed to support a
        large portion of Unicode, are large.  Populating their complete
        outlines can consume substantial amounts of system and video
        memory.  Many applications on a system are likely to access
        the same collections of fonts.  Having fonts loaded by name
        allows GL implementations to coordinate the efficient sharing
        of font outline data among multiple GL application instances.
        This font sharing can have a substantial reduction in the total
        system resources devoted to font data which is not possible if
        the GL is unable to be aware of duplicated font outline data
        within the system.

        Font formats change and evolve over time.  Building font format
        knowledge into applications will ultimately be limiting long-term.

        Fonts are really properly thought of as system resources.
        They represent intellectual property that is typically licensed
        on a per-system basis.  Building font access into the GL promotes
        use of the system's properly licensed fonts.  Most applications
        do not want to be encumbered by licensing issues associated with
        fonts so to the extent that the API makes access to system fonts
        easier, that promotes properly licensed use of fonts.

    37. What is the typographical philosophy for this extension?

        RESOLVED:  This extension relies on other standards to provide
        its typographic backbone and philosophy.

        The character set supported depends on the Unicode standard.

        Specific font formats supported depend on the system but the
        expectation is that standard TrueType, PostScript, and OpenType
        fonts can be used through this extension.  The metrics from such
        fonts will generally be "passed through" the glGetPathMetricsNV
        query.

        The naming of fonts is consistent with the underlying system
        with the expectation that the system's naming is consistent with
        modern web standards for identifying fonts in web content.

        While the specific set of supported fonts may vary from system to
        system based on the available installed fonts, the expectation is
        that standard TrueType fonts such as Arial, New Courier, Georgia,
        etc. will be available on systems that support this extension.

        For applications that demand a set of glyphs that are guaranteed
        to be available, the GL_STANDARD_FONT_NAME font target is
        available for the names "Sans", "Serif", and "Mono" and these
        fonts are understood to match a set of glyphs consistent with the
        DejaVu font set populated with at least the Latin-1 character set.

        The underlying font engine is likely to be FreeType 2 or the
        system's native font engine (such as DirectWrite for newer
        Windows versions).

    38. What is the path rendering philosophy for this extension?

        RESOLVED:  Two-step stencil-based GPU-acceleration + broad-tent
        support for the accepted functionality of path rendering.

        This extension assumes that the two-step "stencil, then cover"
        stencil-based approach to GPU-accelerating path rendering.

        Both stenciling and stroking are supported.  Strokes are
        first-class representations and not treated as fills that
        approximate the stroked region.  For pragmatic reasons, cubic
        Bezier segments, conic segments, and partial elliptical
        (non-circular) arcs path segments are assumed to be approximated
        by a sequence of quadratic Bezier path segments that guarantee
        G1 continuity.

        The contrapositive of this approach is an avoidance of schemes
        based on tessellation of path outlines.

        Paths are defined using both cubic and quadratic Bezier curves.
        This broadly allows path content from TrueType (based on quadratic
        Bezier curves) and PostScript and its font families (based on
        cubic Bezier curves) to be supported.

        Arcs are drawn consistent with both SVG (partial elliptical arcs)
        and PostScript (circular arcs and circular tangent arcs).

        The set of stroking options is a union of the stroking features
        of OpenVG, SVG, XML Paper Specification (XPS), PostScript, and
        other standards.  For example, XPS supports dash caps that other
        standards lack.

        The path queries support the key path queries supported by OpenVG.

    39. Should there be an API for assigning path metric information to
        a path object?

        RESOLVED:  No.

        Path metrics are available when a path object is created with
        glPathGlyphsNV or glPathGlyphRangeNV.  In these cases, the font
        supplies the metric data for these path objects.

        It might be useful to allow these metrics to be specified for an
        arbitrary path object.  This way user-defined path objects could
        appear to have metrics available as if they had been specified
        by glPathGlyphsNV or glPathGlyphRangeNV.

        Supporting the specification of path metrics would require new
        API.  Something like glPathMetricsNV perhaps?  Or having parameter
        names for the font metrics supported by glPathParameter{f,i}v?
        The later approach would probably require new tokens and would
        mean glGetPathParameter{f,i}v should support these tokens too.

        Since the metrics are for information purposes only, meaning
        the rendering functionality for paths never involves the metrics
        (unlike other path parameters), it seems odd to allow information
        to be specified just so it can be queried by the application.
        This doesn't feel like essential functionality though its
        absence may be missed by library developers that want to "fake"
        font loaders.

    40. What happens when an input path object to glWeightPathsNV,
        glInterpolatePathsNV contains an arc command when there are two
        or more path objects involved?

        RESOLVED:  An INVALID_OPERATION error is generated.

        In general, arc commands are not "closed" under linear
        combination.  Said another way, the linear combination of two
        or more arcs is not, in general, itself an arc of the same form.

        glCopyPathNV copies outlines for path objects containing any
        valid commands including arc commands.

    41. When a path object is created from other existing path objects
        through the glWeightPathsNV, glInterpolatePathsNV, or glCopyPathNV
        commands, where does the new path's parameters come from?

        RESOLVED:  While the path commands are interpolated on a
        command-by-command basis with these commands, the path parameters
        should be copied from the first path object specified.

        So for glWeightPathsNV, glInterpolatePathsNV, and glCopyPathNV,
        the path parameters from the path[0], pathA, and srcPath
        parameters respectively.

    42. How is the glyph metric and kerning information specified for
        a path object created from other existing path objects through the
        glWeightPathsNV, glInterpolatePathsNV, or glCopyPathNV commands,
        where does the new path's parameters come from?

        RESOLVED:  The path metric information is set to negative one
        for glWeightPathsNV and glInterpolatePathsNV.

        There's no reasonable way to weight the metric information.
        Metric information is tuned to a particular glyph.

        More explicitly, the path metric information from the first path
        object to be combined is NOT copied (as the parameters are).

        However glCopyPathNV does copy the glyph metric and kerning
        information (since only one path object is involved so there's
        no combination of outlines).

    43. Should there be a way to specify different stroking parameters
        (stroke width, end caps, etc.) within the command sequence of
        a path?

        RESOLVED:  No.

        Existing path rendering standards keep the stroking parameters
        constant for a given path's outline.  For example, there's not
        support for a dashed stroked segment of width 5.0 as well as a
        non-dashed stroked segment with width 9.4 in the same path.

        This wouldn't be impossible to support; commands that changed
        stroking parameters could be supported within the command
        sequence.  However it would complicate the meaning of the path
        parameters for stroking; these parameters could be considered
        defaults for stroking parameters if stroking parameters are not
        otherwise specified.  There's also the complication of when
        new stroking parameters would latch into place.  Would it be
        immediately (mid path?) or not latch until the next "moveto"
        command?  And how would such commands be weighted/interpolated?

        Attempting to support changing stroking parameters within a path
        appears to open up a complicated can of worms.

        The same rendering effect can be achieved with the
        gl{Stencil,Cover}StrokePathInstancedNV commands using multiple
        path object, each with the appropriate stroking parameters for
        the appropriate path segments.

    44. What should the query token for the path color and texture
        coordinate generation coefficients be named?

        RESOLVED:  GL_PATH_GEN_COEFF_NV.

        Alternatively this could be GL_PATH_GEN_COEFF_NV (plural),
        but that doesn't match the precedent set by GL_COEFF used by
        glGetMap{f,d}v.  These existing queries return a plurality of
        coefficients too.

    45. What should the number of coefficients returned when querying the
        path color and texture coordinate generation coefficients depend
        on the current path color or texture coordinate generation mode or
        should a fixed maximum number of coefficients always be returned?

        RESOLVED:  A fixed maximum of 16 coefficients should always
        be returned.

        It is error-prone and likely to result in obscure buffer
        overflow cases if the number of coefficients returned depends
        on the respective current path generation mode.  It is better
        to simply always return 16 values.  Unused coefficients by the
        current generation mode should always be returned as zero.

    46. How does glGetPathLengthNV compare to OpenVG's vgPathLength?

        RESOLVED:  glGetPathLengthNV and vgPathLength compute
        essentially the same result except glGetPathLengthNV returns
        0 when /numSegments/ is 0 whereas vgPathLength considers this
        case an error.

    47. Where does all the discussion of partial elliptical arc
        parameterization come from?

        RESOLVED:  This discussion is based on and fully consistent with:

            http://www.w3.org/TR/SVG/implnote.html#ArcImplementationNotes

    48. Where does the parameterization of the
        GL_CIRCULAR_TANGENT_ARC_TO_NV come from?

        RESOLVED:  The GL_CIRCULAR_TANGENT_ARC_TO_NV is based on the
        PostScript arct command (which is based on arcto) for user paths.

        See the gs_arcto routine in:

            http://svn.ghostscript.com/ghostscript/trunk/gs/base/gspath1.c

    49. How should fog coordinate generation work for path rendering?

        RESOLVED:  The glPathFogGenNV command controls how generation
        of the fog coordinate operates for path rendering commands.

        The GL_FOG enable is tricky because it controls both per-vertex and
        per-fragment processing state (unlike per-vertex lighting and texture
        coordinate generation).

        Simply using the existing fixed-function fog coordinate state is
        undesirable because that 1) entangles fog coordinate generation
        with conventional vertex processing and path vertex processing,
        and 2) the NV_fog_distance extension allows a non-linear fog
        coordinate to be generated through the GL_EYE_RADIAL_NV mode.

        The fog coordinate generation for path rendering can either
        use the fog coordinate "as is" for the entire covered path or
        have the fog coordinate be the negated perspective-divided
        eye-space Z component (which can vary, but only linearly).

    50. What should glyph metrics return for path objects not specified
        by glPathGlyphsNV or glPathGlyphRangeNV?

        RESOLVED:  All queried metrics should return the value -1.

        Negative values are out-of-range for many of the metric values so
        negative values provide a reliable indicator that a path object
        was not specified from a glyph.

    51. How should the fill mode state of path objects created from
        glyphs be initialized?

        RESOLVED:  The initial GL_PATH_FILL_MODE_NV for path objects
        created from glyphs depends on the source font's convention.

        Typically TrueType and newer (all?) PostScript fonts depend on the
        non-zero fill rule.  TrueType fonts assume a clockwise outline
        winding (hence will use GL_COUNT_DOWN_NV) while PostScript
        fonts assume a counterclockwise outline winding (hence will
        use GL_COUNT_UP_NV).

        It's unlikely an actual font will use GL_INVERT as its
        GL_PATH_FILL_MODE_NV but the possibility is allowed.

    52. Should other path object parameters other than the fill mode be
        initialized specially when path objects are specified from glyphs?

        RESOLVED:  No.

        In theory, other path parameters such as stroke width, join style,
        etc. could all be specified from the font.  In practice, most
        font forms don't provide such parameters.

        At least one font format, Bitstream's PFA format, does provide
        such information though how applicable these parameters are to
        a path object is unclear.  The availability of these parameters
        appears to be intended as a way to bold or otherwise dilate the
        glyph's outline rather than being intended for stroking.

        SVG supports stroking of fonts but the stroke-width tag is
        specified in the current user coordinate system rather than
        depending on the particular font or its glyphs.

    53. How should the integers passed to glPathGlyphsNV and
        glPathGlyphRangeNV be mapped to actual glyph outlines for a font?

        RESOLVED:  The integers that come from the charcode array or
        the firstGlyph to firstGlyph+numGlyphs-1 range are treated as
        Unicode character codes if the font has a meaningful mapping of
        Unicode to its glyphs.

        The existence of a meaningful mapping from Unicode to glyph
        outlines is the expected situation.  For fonts without a
        meaningful mapping to Unicode character codes (such as custom
        symbol fonts), the font's standard mapping of character codes to
        glyphs should be used.  This situation should be rare, probably
        due to a font that is poorly authored, very old, or custom built.

    54. How are typographical situations such as ligatures, composite
        characters, glyph substitution, and language-dependent character
        sequence conversion handled?

        RESOLVED:  If a particular behavior is desired for how such
        situations are handled, that is up to the application software
        using this extension.

        For example, in the case of ligatures, multiple Unicode characters
        may map to a single ligature glyph.  Support for ligatures is
        a stylistic typographic decision and the application is free to
        handle this in any of a number of ways; this extension neither
        forces nor precludes specific approaches to handle ligatures.
        The application can overlap existing glyphs to create the
        appearance of a path object by rendering the individual multiple
        Unicode characters overlapped; a ligature character that is
        part of the Unicode character set could be selected; or the
        application could create its own custom path object in this
        situation and render it.

        For composite characters, the underlying font engine used to
        implement this extension may construct composite characters.
        Or this may be a situation where, due to limitations of the
        font or font engine, possibly in combination, this is treated as
        an unknown or missing character where implementation-dependent
        handling is possible.  Such a situation could also exist for a
        ligature character specified by Unicode.

        In general, higher level details of text presentation such
        as ligatures, composite characters, glyph substitution, and
        language-dependent character sequence conversion are beyond the
        scope of this extension.

        See the Unicode FAQ on "Ligatures, Digraphs and Presentation
        Forms":

            http://www.unicode.org/faq/ligature_digraph.html

        In complicated typographical situations, the assumption is that
        the application will construct the appropriate inter-glyph
        transformation values (the transformValues and transformType
        for glStencilFillPathInstancedNV and glCoverFillPathInstancedNV)
        and build digraphs or other presentation forms.

    55. Are relative path commands converted to absolute commands upon
        path specification?

        RESOLVED:  No, relative commands are first-class and are
        maintained as relative commands.

        This includes when relative commands are created by copying,
        interpolating, or weighting existing path objects.  Relative path
        commands must match identical relative path commands and their
        relatively control points are weighted as relative position
        offsets.

        Another implication if this is that if an application modifying
        the control points with glPathSubCoordsNV, those edits can effect
        the outline of subsequent relative commands that depend on the
        modified coordinates.

        The same applies to changing commands.  Editing commands with
        glPathSubCommandsNV can change how coordinates are interpreted
        for the edited commands and subsequent relative commands.

        In other words, if a path object is modified or edited, the
        outline of the path is the same as if the path object had been
        specified from scratch with the same command and coordinate
        sequences.

    56. What does this extension do with so-called "hinting" in outline
        font glyphs?

        RESOLVED:  When a path object is specified from the glyph of a
        font, the path object's outline is specified from the "ideal"
        resolution-independent form of the glyph.

        This is because a path object is rendered (stenciled or covered)
        from a resolution-independent form.  There is an implicit
        assumption in the specified transformation and rendering process
        that the process is unaware of the device coordinate grid.

        This means there's not the knowledge of device coordinate space
        necessary to apply hinting information.

        In TrueType terms, this amounts to the path object's outline for
        a TrueType glyph being the glyph's "master outline".  This means
        the TrueType instructions associated with the glyph are ignored
        and not executed.

        While it is beyond the scope of this extension, there's nothing
        in this extension that keeps an application in decoding itself the
        TrueType master outline of a glyph and performing the grid-fitted
        outline generation at a given arbitrary device resolution.
        Then this fitted outline could be specified for a path object.
        The key observation is that doing so makes the resulting outline
        resolution-dependent which obviates much of the advantage of
        this extension's ability to render from a resolution-independent
        outline.

        Rather than relying on hinting for legibility, applications using
        this extension are likely to rely on multisampling or multiple
        jittered rendering passes for antialiasing and assume a certain
        amount of grayscale appearance as a consequence.

    57. If a font format has bitmap font data, is that used?

        RESOLVED:  No, only resolution-independent outline data is used;
        bitmap data is ignored.  Bitmap-only font formats won't be loaded.

        In the FreeType 2 API, the information available is comparable
        to calling FT_Load_Glyph with the FT_LOAD_NO_SCALE and
        FT_LOAD_NO_BITMAP flags specified.

    58. How is antialiasing of path object rendering accomplished?

        RESOLVED:  Multisampling is the expected way that antialiasing
        will be accomplished when rendering path objects.

        Recall in multisampling that the stencil buffer is maintained
        at per-sample resolution.  This means the coverage determined
        by stenciling path objects should be accurate to the sample
        resolution.

        If a multisampled framebuffer provides N samples per pixel, that
        means that there are N+1 possible coverage weightings of a given
        path with respect to that pixel, assuming a single "stencil, then
        cover step", equal weighting of samples in the final pixel color,
        and the samples for a given pixel belonging to a single pixel.

        One explicit goal of this extension is to maintain a separation
        between coverage and opacity.  The two concepts are often
        conflated treating both as percentages and then modulating
        opacity with coverage.  Conflating the two leads to coverage
        bleeding at what should be sharp, though transparent, edges and
        corners.

        In this extension, the stencil buffer maintains coverage and
        the alpha channel for RGBA colors, which is per-sample when the
        framebuffer format supports multisampling, maintains opacity.

        Philosophically this extension provides a robust and accurate
        mechanism for determining point-sampled coverage for arbitrary
        filled and stroked paths.  The extension does not rely on, nor
        does it even attempt, to compute or approximate a path's area
        coverage with respect to a pixel.  For practical reasons, such
        analytical computations are inevitably approximations for
        arbitrary paths and are difficult to make robust.

        Point sampling of path object rasterization can offer more
        robustness and precision.  Point sampling also allows this
        extension's rendering results to seamlessly co-exist with OpenGL's
        conventional point, line, and polygon rasterization approaches
        which are point-sampled.

        The implication of this observation is path rendered content can
        be mixed with arbitrary OpenGL 3D content, whether rendered with
        depth testing or not.  This provides the very powerful ability to
        mix path rendered and 3D rendered content in the same framebuffer
        in predictable ways with negligible overhead for doing so.

        Keep in mind that 2D path rendered content is transformed by the
        projective modelview-projection transform, just like other OpenGL
        rendering primitives, so fragments generated with path rendering
        have varying depth values that can be depth tested, fogged, etc.

        Point sampling is prone to missing coverage but avoids indicating
        coverage where no actual coverage exists.

        This extension implicitly assumes that GPUs have some maximum
        sample location precision while rasterizing.  This is an artifact
        of subpixel precision.  This concept is built into OpenGL; see
        the GL_SUBPIXEL_BITS implementation-dependent limit.  Developers
        should not expect any additional sampling precision beyond this
        limit.  To get beyond this limit, applications would be expected
        to render at a larger framebuffer resolution and downsample to
        the appropriate resolution or render in some tiled fashion.

        If multisampling provides insufficient antialiasing, further
        antialiasing is possible by rendering with multiple passes.

        For example, applications can use accumulation buffer techniques
        with sub-pixel jittered re-rendering of the entire scene
        to improve the overall quality.  This provides full-scene
        antialiasing.

        Alternatively, a path object itself needing extra antialiasing,
        perhaps because the application has determined the path object
        maps to a small region of the framebuffer in window space, can
        be rendered multiple times, each time with subpixel jittering.
        By writing just into the non-visible alpha component of the
        framebuffer, a coverage percentage at each color sample can
        be accumulated.  Then a final cover operation can blend this
        coverage information into the visible RGB color channels.

        Despite the multiple passes involved, this approach can still
        be several times faster than CPU path rendering methods because
        of the rendering rate possible through GPU acceleration.

    59. How do the multisample fragment operations interact with path
        rendering?

        RESOLVED:  They are ignored for the "stencil" path rendering
        operations (since only the stencil buffer is updated), and they
        work as specified for the "cover" path rendering operation.

        The coverage determination made during the "cover" path
        rendering operation doesn't reflect the path itself but rather
        the conservative coverage provide by the covering operation.
        For this reason, the coverage mask is conservative, meaning
        samples may be covered that don't actually belong to the filled
        or stenciled region of the path being covered.  And exactly how
        conservative this coverage is depends on the implementation.

        Still the coverage is available and can be used as specified in
        section 4.1.3 ("Multisample Fragment Operations").

        The GL_SAMPLE_COVERAGE mode would be more useful if the stencil
        testing was performed prior to the shading of the covered geometry
        and the covered sample mask reflected any discards performed by
        the stencil (or depth) tests.

        The NV_explicit_multisample extension and its
        ARB_texture_multisample functionality (standard with OpenGL 3.2)
        provide explicit control of the multisample mask.  This mask is
        respected for path rendering.

    60. Does creating multiple instances of path objects from the same
        glyph in the same font face "waste memory"?  What about copies
        of objects created with glCopyPathNV?

        RESOLVED:  This is an implementation issue, but it is reasonable
        to expect that copies of path objects created with glCopyPathNV
        will share their outline data on a copy-on-write basis.  This is
        true even if a path object is copied and its path parameters
        are modified (but not the path commands and coordinates).

        It is also reasonable to expect that path objects created with
        glPathGlyphsNV may use copies if there are replicated character
        codes.  While glPathGlyphRangeNV isn't subject to replicated
        character codes, if two or more character codes share the same
        glyph, it would be reasonable to expect the implementation might
        share the outline data.

        It's always possible to use glPathSubCommandsNV or
        glPathSubCoordsNV to modify the path commands and/or coordinate
        data so then sharing will have to be broken.

    61. Why does glPathGlyphsNV (and hence glPathGlyphRangeNV as well)
        not disturb path objects that already exist in the range of path
        objects to be created?

        RESOLVED:  This facilitates a strategy for supporting multiple
        font names specified in preferential order.

        An application can do something like:

          GLint firstPathName = glGenPathsNV(256);
          const GLfloat emScale = 2048;
          glPathGlyphRangeNV(firstPathName, GL_SYSTEM_FONT_NAME_NV,
                             "Helvetica", GL_NONE, 0, 256, emScale);
          glPathGlyphRangeNV(firstPathName, GL_SYSTEM_FONT_NAME_NV,
                             "Arial", GL_NONE, 0, 256, emScale);
          glPathGlyphRangeNV(firstPathName, GL_STANDARD_FONT_NAME_NV,
                             "Sans", GL_NONE, 0, 256, emScale);

        This ensures that path object names /firstPathName/ through
        /firstPathName/+255 will be loaded with the glyphs from Helvetica,
        Arial, or the guaranteed-present Sans font face, in that order
        of preference.

        This is consistent with the CSS font-family property used in
        web standards, including SVG.

    62. Why are the angles for the arc path commands specified with
        degrees (instead of radians)?

        RESOLVED:  Using degrees is consistent with OpenGL's existing
        glRotatef, glRotated, and gluPerspective commands.

        Using degrees for angles is also consistent with the conventions
        of the PostScript, SVG, and OpenVG commands upon which the arc
        path commands are based.

        Using degrees (90 degrees, 30 degrees, 45 degrees) also allows
        important angles be represented exactly with integer values.
        This is relevant for compact coordinate formats and paths defined
        by strings.

    63. Should UTF-8 and UTF-16 be supported for arrays of path names?

        RESOLVED:  Yes.

    64. What order should the arguments be listed when a array of
        path objects with typed elements and a base are specified?

        RESOLVED:

        1) sizei count,
        2) enum pathNameType,
        3) const void *paths,
        4) uint pathBase

        The standard OpenGL parameter pattern is count/type/array.
        Examples of this are glDrawElements and glCallLists.
        (More generally the pattern is count/format/type/array.)

        Having the pathBase parameter last matches the precedent set by.
        glDrawElementsBaseVertex where the base vertex value follows
        the list of element indices.  Hence the pattern
        count/type/array/base.

        The basevertex parameter to glDrawElementsBaseVertex is typed
        GLint; the pathBase parameter is typed GLuint.  GLuint makes
        sense to avoid useless signed/unsigned mismatch warnings from
        C compilers when most values passed to pathBase parameters are
        likely to be from GLuint variables.  When GLuint and GLint are
        both 32-bit data types, the choice is not consequential.

        Commands that use this order are glStencilFillPathInstancedNV,
        glStencilStrokePathInstancedNV, glCoverFillPathInstancedNV,
        glCoverStrokePathInstancedNV, glGetPathMetricsNV, and
        glGetPathSpacingNV.

    65. What order should the arguments be listed when a range of
        path objects is specified?

        RESOLVED:

        1) uint firstPath,
        2) sizei count

        The glDeletePathsNV command and GetPathMetricRangeNV query use
        this order.

        glDeleteLists uses this same order.

    66. Where does the UTF-8 and UTF-16 specification language come from?

        See the RFC "UTF-8, a transformation format of ISO 10646":

            http://tools.ietf.org/html/rfc3629

        See the RFC "UTF-16, an encoding of ISO 10646":

            http://tools.ietf.org/html/rfc2781

        The intent of the specification language is to match these RFCs.

    67. How does the GL_BOUNDING_BOX_OF_BOUNDING_BOXES_NV cover mode
        work for glCoverFillPathInstancedNV and glCoverStrokePathInstancedNV?

        RESOLVED:  The command computes the bounding box of all the
        path's bounding boxes.  (This can be too conservative for an
        arbitrarily arranged collection of path objects but works well
        enough for glyphs in line of text.)

        This bounding box has a consistent counterclockwise winding
        order no matter what path objects are listed.  This property
        is a combination of how glRectf works and how the parameters to
        glRectf are computed.

        The object-space z (depth) is always zero.  (This behavior is
        a consequence of the primitive being emitted by glRectf.) The
        matrix elements in the Z row (if such a row exists) of the
        transforms specified for glCoverFillPathInstancedNV and
        glCoverStrokePathInstancedNV is ignored.

        Programmers are cautioned that this could result in the
        covering geometry being view-frustum culled if the programmer
        is not careful when using 3D transformTypes (GL_TRANSLATE_3D_NV,
        GL_AFFINE_3D_NV, GL_TRANSPOSE_AFFINE_3D_NV).  To guard against
        this mishap, consider something such as the following:

            glMatrixPushEXT(GL_PROJECTION);
              glScalef(1,1,0);
              glCoverFillPathInstancedNV(...);
            glMatrixPopEXT(GL_PROJECTION);

        This essentially forces the clip-space Z to be zero which will
        never be clipped by the near or far view-frustum clip planes.

        If depth testing is desired, perform the depth testing during the
        "stenciling" step so that depth testing is unnecessary during the
        "covering" step done by the glCoverFillPathInstancedNV command.

    68. What happens when the radius of a circular arc command is
        negative?

        UNRESOLVED:  The intent is to match the behavior of the PostScript
        circular arc commands (arc, arcn, arct).  Unfortunately the
        PostScript specification is not entirely clear about how negative
        radius is handled.

        Table 5.arcParameterSpecialization has absolute values (abs)
        computed for the rv and rh columns.

        However, the points A and B (used for arc and arcn) are computed
        with c[2] directly (without an absolute value).

        This computation looks consistent with Ghostscript's behavior
        for arct:

            dist = abs(c[4] * num/denom)
            l0 = dist/sqrt(dot(d0,d0)) * c[4]/abs(c[4])
            l2 = dist/sqrt(dot(d2,d2)) * c[4]/abs(c[4])

        Could this simply be:

            dist = c[4] * num/denom
            l0 = dist/sqrt(dot(d0,d0))
            l2 = dist/sqrt(dot(d2,d2))

       Probably.

       This really needs testing and comparison with a PostScript
       implementation to make sure the specified equations really match
       PostScript's implemented behavior.

    69. What happens when the two angles (c[2] and c[3]) for a circular
        arc command (GL_CIRCULAR_CCW_ARC_TO_NV or
        GL_CIRCULAR_CW_ARC_TO_NV) create 1 or more full revolutions?

        UNRESOLVED:  The intent is to match the behavior of the PostScript
        circular arc commands (arc and arcn).

        PostScript specifies that "If angle2 is less than angle1, it is
        increased by multiples of 360 [degrees] until it becomes greater
        than or equal to angle1.  No other adjustments are made to the
        two angles.  In particular, if the difference angle2-angle1
        exceeds 360 [degrees], the resulting path will trace portions
        of the circle more than once."

        The current equations based on an end-point partial elliptical arc
        parameterization achieve this.  Extra parametric behavior would be
        necessary to trace a circle multiple times.  The current equations
        in Table 5.pathEquations do not capture this (but should).

        This needs to be thought through carefully to make sure stroking,
        particularly when dashed, is handled correctly.

    70. PostScript generates a limitcheck error when numbers are
        encountered that exceed the implementation limit for real numbers.
        Should the PostScript grammar treat such situations as a parsing
        error?

        RESOLVED:  No, it's not a parsing error, but the results in
        such a situation are likely to be undefined.

        This paragraph in Section 5.X.1 ("Path Specification") applies
        which begins "If a value specified for a coordinate (however
        the coordinate is specified) or a value computed from these
        coordinates (as specified in the discussion that follows)
        exceeds the implementation's maximum representable value for a
        single-precision floating-point number, ..."

        The PostScript's notion of a limitcheck error doesn't nicely
        correspond to a parsing error.  And PostScript's notion of "the
        implementation limit for real numbers" (likely double precision)
        might not correspond to the GL's notion of floating-point
        (typically single precision).

        The PostScript notion of a limitcheck on numeric range is
        particularly hard to enforce with relative commands where the
        limitcheck might not occur until all the relative offsets are
        applied, something which isn't really part of parsing.

        What an actual implementation does may vary but a likely
        implementation approach is generate an IEEE infinity value when
        single-precision floating-point range is exceeded.  This will
        generate undefined rendering behavior.

        SVG doesn't offer guidance in its specification when coordinate
        values exceed the representable range of floating-point.
        Presumably such range overflows result in implementation-dependent
        undefined rendering behavior too.

    71. What happens when the radius of a OpenVG-style partial elliptical
        arc commands is negative?

        RESOLVED:  The absolute value of the radius is used for
        the OpenVG-style arc commands GL_SMALL_CCW_ARC_TO_NV,
        GL_RELATIVE_SMALL_CCW_ARC_TO_NV, GL_SMALL_CW_ARC_TO_NV,
        GL_RELATIVE_SMALL_CW_ARC_TO_NV, GL_LARGE_CCW_ARC_TO_NV,
        GL_RELATIVE_LARGE_CCW_ARC_TO_NV, GL_LARGE_CW_ARC_TO_NV, and
        GL_RELATIVE_SMALL_CW_ARC_TO_NV.

        Table 5.arcParameterSpecialization specifies an absolute value
        (abs) in the rh and rv entries of all these commands.

        The OpenVG specification is clear on this point in section 8.4
        ("Elliptical Arcs") saying "Negative values of [radii] rh and
        rv are replaced with their absolute values."

    72. What should happen for a stroked subpath that is zero length?

        UNRESOLVED:  Not sure yet.

        SVG gives this advice:

            http://www.w3.org/TR/SVG11/implnote.html#PathElementImplementationNotes

        Probably need to check what other path renders, particularly
        PostScript do in this situation.  Requires testing actual
        implementations because the specifications are not clear.

    73. Why have the GL_PATH_CLIENT_LENGTH_NV path parameter?

        RESOLVED:  This supports SVG's pathLength attribute used to
        calibrate distance-along-a-path computations.

        This applies to dashing a stroked segment, but does NOT
        apply to the lengths returned by the glGetPathLengthNV and
        glPointAlongPathNV queries.

        The client length just applies to dashing because having a
        client length that is different from the GL's computed length
        for a path may greatly affect the dashing pattern.  The client
        knows the path's client length, but the GL doesn't unless the
        client state is available to the GL when dashing a stroked path.

        It's better to have the client send the client path length
        unconditionally than require the client to query the GL's computed
        path length ahead of any sending of a rescaled version of the
        dash offset or dash array.

        For the queries, presumably the client can perform the necessary
        scaling by the client length itself if that's desirable.

    74. Should there be a query for GL_PATH_END_CAPS_NV and
        GL_PATH_DASH_CAPS_NV?

        RESOLVED:  No.  You have to query GL_PATH_INTIAL_END_CAP_NV or
        GL_PATH_TERMINAL_END_CAP_NV for the each respective end cap; or
        query GL_PATH_INITIAL_DASH_CAP_NV or GL_PATH_TERMINAL_DASH_CAP_NV
        for each respective dash cap.

        GL_PATH_END_CAPS_NV and GL_PATH_DASH_CAPS_NV are convenient
        for most path rendering systems that have identical initial
        and terminal end and dash caps, but are NOT supported by
        glGetPathParameteriv or glGetPathParameterfv.

    75. What should the path format tokens for SVG and PostScript tokens
        be named?

        RESOLVED:  Use the abbreviated names SVG and PS respectively:
        GL_PATH_FORMAT_SVG_NV and GL_PATH_FORMAT_PS_NV.  These names
        are shorter and avoid putting an Adobe trademark in a token name.

        Future extensions might want to add version numbers to these
        abbreviated names (another reason to stick with short abbreviated
        names).

    76. In what content (GL client or GL server) are font file names
        and system font names interpreted?

        RESOLVED:  The GL_STANDARD_FONT_NAME_NV and GL_SYSTEM_FONT_NAME_NV
        font targets map their respective font names to a font within
        the GL server.  The GL_FILE_NAME_NV font target does the file
        reading in the GL client; for GLX, there needs to be GLX protocol
        to transfer glyphs including their kerning and metric data to
        the GL server.

    77. How can the glPathSubCommandsNV command be used to append to
        the end of an existing path object?

        RESOLVED:  If you set the /commandStart/ parameter to
        glPathSubCommandsNV to be sufficiently large (greater or equal
        to the number of path commands in the path object suffices),
        that works to append commands.

    78. Does depth offset (a.k.a. polygon offset) work when using the
        "stencil" and "cover" path operations?

        RESOLVED:  Yes with caveats.

        The "stencil" path operations use the
        GL_PATH_STENCIL_DEPTH_OFFSET_FACTOR_NV and
        GL_PATH_STENCIL_DEPTH_OFFSET_UNITS_NV state set by
        glPathStencilDepthOffsetNV.  There is no specific enable; instead
        set the scale and units to zero if no depth offset is desired.

        The "cover" path operations use the polygon depth offset state if
        the GL_POLYGON_OFFSET_FILL enable is enabled, using the polygon
        offset factor and units specified for glPolygonOffset.  This is
        because the "cover" operation (unlike the stencil operation)
        does rasterize a polygon primitive.

        Depth offset is useful when a path rendered decal is applied
        on depth tested 3D geometry and the path rendered geometry has
        to be biased forward (negative bias) by polygon offset to avoid
        depth ambiguities.  See issue #120 for details.

        This is also useful when putting path rendered primitives into
        shadow maps with a positive depth bias to avoid shadow acne
        issues.

        There is NOT a guarantee that the depth offset computed for a
        "stencil" operation will exactly match the depth offset for a
        "cover" operation given identical path object and transformations.
        The two offsets will be close but not generally exact for all
        generated samples.

    79. Can fragment shaders access the facingness state during a cover
        operation?

        RESOLVED:  Yes, the gl_FrontFacing special variable in GLSL
        is available.  So is the fragment.facing fragment attribute
        binding in NV_fragment_program2 and subsequent NVIDIA shader
        assembly extensions.

        In cases where the path rendered primitive is "very edge" on the
        facingness fragment state may be ambiguous in extreme situations.

    80. When are various computed path parameters re-computed?

        RESOLVED:  If the computed parameter parameters
        (PATH_COMMAND_COUNT_NV, PATH_COORD_COUNT_NV,
        PATH_DASH_ARRAY_COUNT_NV, PATH_COMPUTED_LENGTH_NV,
        PATH_OBJECT_BOUNDING_BOX_NV, PATH_FILL_BOUNDING_BOX_NV, and
        PATH_STROKE_BOUNDING_BOX_NV) are queried, the values returned
        always reflect the most up-to-date state of the path object.

        This also includes when path object parameters are used in
        contexts such as instanced "cover" operations.

    81. Should projective 2D path coordinates be supported?

        RESOLVED:  No.  Major path rendering standards don't support
        projective 2D path coordinates.

        Moreover, projective 2D path coordinates create technical
        problems because the projective transformation of projective
        2D path coordinates for cubic Bezier curves do not necessarily
        retain their topology (serpentine, cusp, or loop).

    82. Should a non-dashed stroked path's coverage be the same
        independent of how its control points are specified?

        RESOLVED:  Yes, this is a symmetry rule mandated by the OpenXML
        Paper Specification.  This applies to lines and Bezier curves.

        So a cubic Bezier curve defined by control points cp0, cp1, cp2,
        and cp3 should generate the same stroked coverage (assuming the
        same stroke parameters and requiring the dash array count to be
        zero) as a cubic Bezier curve with control points cp3, cp2, cp1,
        and cp0 (so the reversed control point order).

        XXX Unresolved if it applies to arcs.

    83. Should character aliases used to specify path commands be returned
        as their character alias values or remapped to the actual token
        name of the command?

        RESOLVED:  Remapped.  Any path commands specified with a
        character alias value (from Table 5.pathCommands) is returned
        as the command's token value instead.

        This avoids applications calling glGetPathCommandsNV from having
        bugs where they handle token names but not character aliases.

        This also makes it simpler to say "identical" when saying command
        sequences must match for glWeightPathsNV.  Character aliases
        remapped to command token values makes it unambiguous that
        GL_LINE_TO and 'L" are the identical command.

    84. Is there a way to use this extension to trade-off rendering performance
        for more effective samples per pixel to improve coverage quality?

        RESOLVED:  Yes.

        This code demonstrates how multiple passes could accumulate
        coverage information in the alpha channel of the framebuffer and
        then a final cover pass could blend the incoming color with the
        accumulated coverage from the framebuffer's alpha channel.

            // INITIALIZATION
            // assume stencil is cleared to zero and framebuffer alpha is clear to zero
            const int coveragePassesToAccumulate = 4;
            glEnable(GL_STENCIL_TEST);
            glStencilFunc(GL_NOT_EQUAL, 0x80, 0x7F);
            glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);  // tricky: zero 0x7F mask stencil on covers, but set 0x80
            glColorMask(0,0,0,1);  // just update alpha

            // M STENCIL+COVER PASSES to accumulate jittered path coverage into framebuffer's alpha channel
            glStencilFillPathNV(path, GL_COUNT_UP_NV, 0x7F);
            glCoverFillPathNV(path, GL_PATH_FILL_COVER_MODE_NV);
            glEnable(GL_BLEND);
            glBlendFunc(GL_ONE, GL_ONE); // sum up alpha
            glColor4f(0,0,0, 1.0/coveragePassesToAccumulate );
            static const GLfloat jitters[4][2] = {
              {0,0},  /* various small subpixel jitter X & Y values */
            };
            for (i=1; i<coveragePassesToAccumulate ; i++) {
              glMatrixPushEXT(GL_MODELVIEW); {
                glMatrixTranslatef(GL_MODELVIEW, jitters[i][0], jitters[i][1], 0);
                glStencilFillPathNV(path, GL_COUNT_UP_NV, 0x7F);
                glCoverFillPathNV(path, GL_PATH_FILL_COVER_MODE_NV);
              } glMatrixPopEXT(GL_MODELVIEW);
            }

            // FINAL COVER PASS uses accumulated coverage stashed in destination alpha
            glColorMask(1,1,1,1);
            // modulate RGB with destination alpha and then zero destination alpha
            glBlendFuncSeparate(GL_DST_ALPHA, GL_ZERO, GL_ZERO, GL_ZERO);
            glColor4f(red, green, blue, dontcare);  // some color
            glStencilFunc(GL_EQUAL, 0x80, 0xFF);  // update any sample touched in earlier passes
            glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);  // now set stencil back to zero (clearing 0x80)
            glCoverFillPathInstancedNV(coveragePassesToAccumulate,
                GL_UNSIGNED_BYTE, "\0\0\0\0",  // tricky: draw path objects path+0,path+0,path+0,path+0
                path,  // this is that path object that is added to zero four times
                GL_BOUNDING_BOX_OF_BOUNDING_BOXES_NV, GL_TRANSLATE_2D_NV, jitters);

        Assuming N passes and M samples per pixel, this approach
        accumulates coverage for N*M+1 grayscale levels doing N stencil
        operations and N+1 cover operations.

    85. Why do the commands glGenPathsNV and glDeletePathsNV allocate
        contiguous ranges of path objects instead of returning an array of
        (possibly scattered) names and deleting an array of names?

        RESOLVED:  The expectation that path objects will be arranged
        as characters mapping to glyphs warrants adopting the object
        model of display lists.

        glPathGlyphRangeNV, the instanced commands, and the metric and
        kerning queries all rely on this assumption.

    86. What do the stencil and cover commands do if the specified path
        name does not refer to an existing path object?

        RESOLVED:  Do nothing (and generate no error).

        This is useful to avoid rendering unpopulated path objects.

        This "do nothing" behavior also applies to the instanced
        stencil and cover routines that are expressed in terms of
        glStencilFillPathNV, glStencilStrokePathNV, glCoverFillPathNV,
        and glCoverStrokePathNV.

        Applications that want some "glyph is missing" for non-existent
        path objects can use glCopyPathNV to copy some existing path
        object's "glyph is missing" outline to non-existent paths.
        Alternatively, glPathGlyphRangeNV (or glPathGlyphsNV) can be used
        with the GL_STANDARD_FONT_NAME of "Missing" to populate a range
        (or sequence) of path objects with a standard missing glyph
        (typically a rectangle).  (See issue #89.)

        Queries (glGetPathParameteriv, etc.) allowing only a single path
        object to be specified, generate a GL_INVALID_OPERATION error
        if the path name does not exist.

        glWeightPaths, glInterpolatePathsNV, and glCopyPathNV generate
        a GL_INVALID_OPERATION error if any of the named source paths
        do not exist.

        Path commands that modify the commands, coordinates, or parameters
        of existing path objects (as opposed to specifying a path object
        completely) generate a GL_INVALID_OPERATION error if the path
        name does not exist.

    87. What of this extension's per-context state should apply to
        glPushAttrib and glPopAttrib?

        RESOLVED:  Apply the existing conventions; see new table 6.X,
        "Path (state per context)".

        The path fog generation mode applies to GL_FOG_BIT.

        The path color generation mode and coefficients apply to
        GL_LIGHTING_BIT.

        The path texture coordinate set generation modes and coefficients
        apply to GL_TEXTURE_BIT.

        The path error position is not pushed or popped, following the
        convention of the ARB_vertex_program extension.

    88. How should the numCoords parameter to the various path
        specification commands work?

        RESOLVED:  When there is also a /numCoords/ parameter,
        the GL_INVALID_OPERATION error is generated if the number of
        coordinates is not equal to the number of coordinates needed by
        the command's specified path command sequence.  This provides
        a sanity check.

        For the glPathSubCoordsNV command, there's no requirement that
        the range of coordinates "match up" with path command boundaries
        for coordinates.

        Some consideration was given to specifically treating a value of
        zero for /numCoords/ by allowing the number of coordinates to
        be based on the command's corresponding path command sequence.
        This would allow an application to bypass the sanity check if
        the application didn't exactly know how many coordinates a path
        command sequence required.  This was rejected because it is likely
        error-prone and it means when such commands are compiled into
        a display list or packed into GLX protocol, the path command
        sequence would then have the be scanned.  This would make the
        GL client unnecessarily knowledgeable about the supported path
        commands.  So for both "safety" and implementation reasons,
        the /numCoords/ value of zero is not specially interpreted;
        it means that the path command sequence really is expected to
        require zero coordinates (not generally the case except for the
        GL_CLOSE_PATH_NV command).

    89. How can an application guarantee that every glyph in a range
        of Unicode character codes has /some/ default outline defined?

        RESOLVED:  The "Missing" name with the GL_STANDARD_FONT_NAME_NV
        target populates the entire sequence or range of path objects
        with a default outlines.

        Example:

          const int allOfUnicode = 1<<21;  // Entire 21-bit Unicode range!
          const GLfloat emScale = 2048;
          GLuint glyphBase = glGenPathsNV(allOfUnicode);
          glPathGlyphRangeNV(glyphBase,
                             GL_STANDARD_FONT_NAME_NV, "Missing", GL_NONE,
                             0, allOfUnicode, emScale);

    90. Does the "Missing" font name used for GL_STANDARD_FONT_NAME_NV
        skip or avoid character codes that Unicode designates as white
        space, line terminators, etc.

        RESOLVED:  The "Missing" font name populates these with a path
        object with the missing outline (a box) and corresponding metrics
        for such spacing characters and all other characters.

        The assumption is that Unicode fonts will populate various
        blank space, line and paragraph terminators, and other blank
        or ignorable character points with appropriate null outlines
        and zero-width metrics.  Because glPathGlyphRangeNV and
        glPathGlyphsNV don't re-specify existing path objects, these
        will be left along if the "Missing" standard font is the last
        font used to specify a given range of path objects.  All white
        space and separator character codes less than 256 (the Latin-1)
        range will be populated by the "Serif", "Sans", and "Mono"
        standard fonts if specified because these guarantee the Latin-1
        range is populated.  These are specifically U+0009..000D, U+0020,
        U+0085, 0x00A0 but also other blank or ignorable character points
        such as control characters.

        For more information:

            http://unicode.org/faq/unsup_char.html

    91. What is the /emScale/ parameter to glPathGlyphRangeNV and
        glPathGlyphsNV for and how should it be used?

        RESOLVED:  /emScale/ exists to ensure multiple fonts, potentially
        with distinct font unit scales, can be scaled to a consistent
        scale.

        Typically TrueType fonts are authored for 2048 font units per
        Em while Type1 fonts have been historically authored to 1000
        font units per Em.

        Typically a good value for /emScale/ is 2048 to match the
        convention of TrueType.  This avoid TrueType font metrics from
        being rescaled.

        Setting /emScale/ to zero allows the native font units per Em
        to be used.  Be careful because outlines of path objects for
        glyphs from fonts with different font units per Em will have
        different scales.

    92. Should glWeightPathsNV work for a single path object?

        RESOLVED:  No, two or more paths are required to generate a
        weighted path.

        Use glCopyPathNV if copying a single path object is desired.

        glCopyPathNV copies glyph metrics and kerning information and
        allows arc commands while glWeightPathsNV does not.

        While glInterpolatePathsNV can be expressed in terms of
        glWeightPathsNV, glCopyPathNV cannot.

    94. What should the initial join style of a path object be?

        RESOLVED:  GL_MITER_REVERT_NV.  This is consistent with the join
        style used by SVG, PostScript, PDF, and Cairo.

        Note that Flash, XPS, and Qt use GL_MITER_TRUNCATE_NV instead.

        Arguably GL_MITER_TRUNCATE_NV is a "nicer" join style because
        the miter does not "pop" from miter to bevel when the miter limit
        is exceeded; instead when the miter limit is approached and then
        exceeded, the miter stops growing further and simply loses its
        sharp tip.

    95. What type of triangle should the GL_TRIANGULAR cap be?

        RESOLVED:  A right triangle.

        This is consistent with the XPS specification.  Other standards
        don't support triangular caps.

    96. Can NV_path_rendering be implemented without *any* dependencies
        on system specific fonts?

        RESOLVED:  YES.

        Say a platform had poor or unstable interfaces for accessing
        system specific fonts (e.g. Linux).  In the case of Linux,
        resolution-independent fonts are typically accessed through a
        combination of freetype2 and fontconfig.

        One or both of these standards may be missing from the platform
        or be unreliable or misconfigured.

        In such a case, NV_path_rendering could be implemented so
        that the GL_SYSTEM_FONT_NAME_NV usage for glPathGlyphsNV and
        glPathGlyphRangeNV would never populate path object names with
        glyphs.

        However the GL_STANDARD_FONT_NAME_NV usage would still be
        guaranteed.  The GL_STANDARD_FONT_NAME_NV usage by providing
        the required set of pre-compiled font outlines built-in into the
        driver directly (using IP unencumbered font outlines such as
        the DejaVu fonts).

        This design means that applications that use the approach
        (copied from the Overview) will work:

            glPathGlyphRangeNV(glyphBase,
                               GL_SYSTEM_FONT_NAME_NV, "Helvetica", GL_BOLD_BIT_NV,
                               0, numChars, emScale);
            glPathGlyphRangeNV(glyphBase,
                               GL_SYSTEM_FONT_NAME_NV, "Arial", GL_BOLD_BIT_NV,
                               0, numChars, emScale);
            glPathGlyphRangeNV(glyphBase,
                               GL_STANDARD_FONT_NAME_NV, "Sans", GL_BOLD_BIT_NV,
                               0, numChars, emScale);

        In this case, the two initial glPathGlyphRangeNV calls
        will fail to populate the range of path objects from
        [glyphBase,glyphBase+numChars-1] but the third call will populate
        the range.

        This allows NV_path_rendering to be implemented with ZERO
        dependencies on the system to provide glyphs from system fonts while
        applications can still utilize fonts in their path rendering.

        While this is an allowed implementation approach, actual
        implementations should make reasonable efforts to provide access
        to system fonts if possible.

    96. What is GL_DASH_OFFSET_RESET_NV for?

        RESOLVED:  OpenVG supports the concept of a "dash phase reset"
        (see VG_DASH_PHASE_RESET) that controls whether or not the dash
        pattern (with its offset) resets at "move to" command boundaries
        within a path's command sequence.

        Rather than use a boolean value (as OpenVG does), the
        GL_DASH_OFFSET_RESET_NV path parameter takes an enumeration
        consisting of GL_MOVE_TO_RESETS_NV and GL_MOVE_TO_CONTINUES_NV
        to be more explicit about how the dash offset reset parameter
        affects the dash pattern.

        Technically, what this specification calls the "dash offset"
        is what OpenVG calls its "dash phase".  This specification
        uses "dash phase" to mean what OpenVG calls "dash phase reset"
        because the word "reset" is built into the GL token values.

        When there is a dash phase reset, the dash offset is set to the
        value of the path's GL_DASH_OFFSET_NV parameter (consistent with
        OpenVG).

    97. What APIs say "dash offset" and what APIs say "dash phase" and
        why does this extension use "dash offset"?

        RESOLVED:  PostScript, PDF, Cairo, Qt, XPS, SVG, and Silverlight
        use the "dash offset" for the offset to the dash pattern, same
        as this extension.

        OpenVG, Quartz 2D, Java 2D, Illustrator, and Skia use the term
        "dash phase" for the identical functionality.

        OFFSET makes more sense in the OpenGL context because lots of
        OpenGL tokens already use OFFSET in their token names.  Core
        OpenGL 4.0 has 16 such tokens already.

    98. What is the motivation for glTransformPathNV?

        RESOLVED:  Sometimes a path should be stroked with a stroke width
        that is constant in a particular space (such as window space).

        Once example of this is SVG 1.2 Tiny's "non-scaling stroke"
        property.  The stroke width is supposed to be maintained after
        transformation into pixel space.

        This could be implemented with this extension by transforming
        the path object into the appropriate space so that the user-space
        stroke width will match the transformed space.

    99. Should the specification say more about how arcs are transformed?

        UNRESOLVED:  Certainly yes, but I'm not sure exactly how to
        specify how arcs are transformed with a suitable level of
        formalism.

        I'm not clear if partial elliptical arcs can be subjected to
        projective transformations and remain partial elliptical arcs.
        I believe they can.

   100. How is glTransformPathNV different from OpenVG's vgTransformPath?

        RESOLVED:  The two commands are similar.

        The OpenVG 1.1 version always converts vertical and
        horizontal line commands to generic line to commands where as
        glTransformPathNV does that only if the resulting transformed
        line segment is no longer either vertical or horizontal in the
        new coordinate system.  This allows cases such as 90 degree
        rotations or scaling without rotation to preserve the compactness
        of vertical and horizontal line segments.

        In OpenVG 1.1, the VG_MATRIX_PATH_USER_TO_SURFACE matrix
        used by vgTransformPath is a 3x3 projective matrix where as
        glTransformPathNV supports up to 4x4 projective matrices.
        Note that the z component of any transformed coordinate is
        effectively discarded by glTransformPathNV so the z row and
        column is not consequential to the resulting transformed path.
        The rationale for this is to allow the same 4x4 transform matrix
        array used by 3D to be used by glTransformPathNV.

        In OpenVG 1.1, the matrix is implicitly supplied by the
        VG_MATRIX_PATH_UER_TO_SURFACE matrix whereas in glTransformPathNV,
        the matrix is explicitly specified.

        XXX Perhaps there should be a special mode that uses the
        modelview-projection-viewport transform implicitly?

   101. Can you provide an example of non-scaling strokes implemented with
        glTransformPathNV?

        UNRESOLVED:  To be written.

   102. What happens if a command that creates a path from existing path
        objects has the result path name as one of the inputs?

        RESOLVED:  This is expected to just work.  The new path object
        is created from the existing ones, then the new path object
        replaces any path object with the resulting path object name.

   103. Should glTransformPathNV support projective transformations?

        RESOLVED:  No, such projective transformations could result
        in path commands transitioning from non-rational to rational
        boundaries.

        If points on the path boundary are generated by non-rational
        boundaries, the resulting transformation, assuming a
        non-projective transformation, also results in non-rational
        boundaries.

   104. Should there be a distinct stencil function state for path
        stenciling?

        RESOLVED:  YES.  glPathStencilFunc sets the state.  How the
        stencil state needs to be configured for path covering is
        different than how the stencil function is configured typically
        for path stenciling.

        For example, stencil covering might use
        glStencilFunc(GL_NOT_EQUAL,0,~0) while path stenciling would
        use GL_ALWAYS for the path stenciling stencil test.

        However there are other situations such as path clipping where it
        is useful to have the path stencil function configured differently
        such as glPathStencilFunc(GL_NOT_EQUAL, 0x00, 0x80) or other
        similar path clipping test.

   105. Is there back- and front-facing path stencil function state?

        RESOLVED:  NO.  There is a single stencil function, reference
        value, and read mask.  The path stenciling operation doesn't
        have a sense of front- and back-facing.

   106. Does the path stencil function state apply always or only if
        stencil testing is enabled?

        RESOLVED:  Always.  If you want to avoid discarding samples
        from this test, use the GL_ALWAYS path stencil function (which
        is the initial context state).

   107. Does the glPathStencilFuncNV state affect the operation of
        the stencil test during path cover operations?

        RESOLVED:  NO, the path stencil state updated by
        glPathStencilFuncNV only affects the path stencil (not cover)
        operations.

        For the path cover operations, the *normal* stencil test applies.
        For the stencil test to apply to path cover operations, the
        stencil test (GL_STENCIL_TEST) must be enabled.

   108. Should path objects be shared among rendering contexts in the
        same manner as display lists and texture objects?

        RESOLVED:  Yes.

        See the "Additions to the AGL/GLX/WGL Specifications" section.

        Because path objects are not "bound" there are stricter
        serialization requirements than for "bindable" objects such as
        programs and textures.

        Due to NVIDIA driver bug 1315267, path objects were not actually
        shared among contexts prior to Driver release 320.xx (July 2013).

   109. Should the kerning separations be 2D offsets or 1D horizontal
        translations?

        UNRESOLVED:  The specification is currently written for 1D
        horizontal translations.

        TrueType fonts appears to provide 1D horizontal translations
        for kerning.

        However a PostScript font can contain 2D offsets for kerning.

        Are 2D offsets really used?  2D offsets seem like an unnecessary
        complication when they are unlikely to be common.

        XXX Need to study this situation further.

        If 99.99% of fonts never use 2D offsets, it is annoying to have
        them for the ultra minority that might.  Not sure what the real
        situation is...

        Perhaps we could provide tokens to query either 1D horizontal
        translations or the more general 2D offsets.  But how would an
        application know whether a font actually specified 2D offsets
        or not??

   110. What is the initial miter limit of a path object?

        RESOLVED:  4 to match the SVG specification.  See:

            http://www.w3.org/TR/SVG/painting.html#StrokeMiterlimitProperty

        There is a lot of variability in initial miter limit values among
        path rendering APIs.  The SVG initial miter limit is chosen
        because SVG is an open, web-based standard.

        For Cairo, the initial miter limit is 10.

        For Direct2D, the initial miter limit is 10.

        For Flash, the initial miter limit is 3.

        For PostScript, the initial miter limit is 10.

        For Qt, the initial miter limit is 2 units of the stroke width.

        For Skia, the initial miter limit is 4.

   111. Should initial path object state such as miter limit, stroke width, etc.
        be determined by "latching" per-context initial value state for
        these parameters?

        UNRESOLVED

        Possibly.  That would make it easier for a particular path
        rendering API's conventions initial conventions be consistently used
        to initialize path object parameters.

   112. Should glPathFogGenNV's GL_FRAGMENT_DEPTH mode provide a
        perspective-divided value?

        RESOLVED:  No, -ze is provided rather than -ze/we.

        Providing -ze/we would not interpolate properly over the path.
        Typically the modelview matrix used to compute ze is affine so we
        will be 1.0 in such cases and the lack of division won't matter.

        If the modelview matrix is projective, the application can choose
        to interpolate we as a texture coordinate with glPathTexGenNV's
        GL_EYE_LINEAR mode and perform the division -ze/we during fragment
        coloring.

   113. How should path color and texture coordinate generation be
        disabled?

        RESOLVED:  For color:

           glPathColorGenNV(colorFormat, GL_NONE, GL_NONE/*colorFormat*/, NULL);

        For texture coordinate generation:

           glPathTexGenNV(GL_TEXTURE_0+i, GL_NONE, 0/*components*/, NULL);

        The coeffs array could be an arbitrary pointer because it will
        only be dereferenced if the genMode is GL_NONE, but NULL is a
        suitable value to document this fact.

        Querying the respective coefficients after path color or texture
        coordinate disabling commands above should return 16 zeros.

   114. How should path color and texture coordinate generation interact
        with the GL_BOUNDING_BOX_OF_BOUNDING_BOXES_NV instanced cover
        mode?

        RESOLVED:  The effective bounding box is the union of all the
        instanced bounding boxes.

        This is useful for something such as a line of text rendered as
        sequence of glyph path objects where the line of text should share
        a common gradient tied to the bounding box of the line of text.

   115. How do I ignore kerning when using glGetPathSpacingNV?

        RESOLVED:  Pass in 0.0 for the kerningScale parameter.

   116. How do I query the raw kerning parameters?

        RESOLVED:  Pass in 0.0 for the advanceScale parameter and 1.0
        for the kerningScale parameter.

   117. Do I need to use GL_TRANSLATE_2D_NV when getting spacing
        information from glGetPathSpacingNV?

        RESOLVED:  Typically GL_TRANSLATE_X_NV is fine.

        Typically most kerned fonts (particularly TrueType fonts) using
        kerning offsets that are horizontal only.  PostScript technically
        allows 2D (x,y) kerning offsets and FreeType 2's FT_Get_Kerning
        API also returns 2D kerning vectors.

        To support 2D kerning vectors, glGetPathSpacingNV accepts
        GL_TRANSLATE_2D_NV as well as GL_TRANSLATE_X_NV.

        For most fonts, the Y offset can be expected to be zero.

   118. Why have the /pathParameterTemplate/ parameter to
        glPathGlyphRangeNV and glPathGlyphsNV?

        RESOLVED:  Path object specified from glyphs often need parameters
        specified on a per-font basis that are distinct from the initial
        path object parameters in table 6.Y.

        Rather than force an application to respecify the path parameters
        of all the path objects in a range of path objects for glyphs,
        it is more efficient for such glyph-initialized path objects
        to simply use parameters from another existing path object as
        a template.

        For example, the default stroke width of 1.0 might need to be
        respecified for every path object corresponding to a range of
        glyphs for rendering stroked glyphs.  If the emScale for a glyph
        is 2048 (typical of TrueType fonts), then 1.0 is too thin to be a
        discernable stroke width.  A value such as 10% of the Em scale (so
        10% of 2048 would be 20.48) is likely to be a more useful value.

        Similarly, GL_ROUND_NV or GL_BEVEL_NV are better join styles
        for stroking glyphs than the standard join style initial value
        GL_MITER_REVERT_NV.

        A shared dash pattern for all path objects belonging to a single
        set of glyphs is much easier to specify from a template path
        object.

   119. Are system font names and file names for fonts case-sensitive?

        RESOLVED:  Standard font names (such as "Mono" and "Missing"
        are case-sensitive).  System font name names and file names for
        fonts should match the system's policy for case-sensitivity of
        font names and file names respectively.

        Linux and other Unix-like operating systems have case-sensitive
        file names.  Windows has case-insensitive file names.  Windows and
        FontConfig-based systems have case-insensitive system font names.

   120. Why have PathStencilDepthOffsetNV and PathCoverDepthFuncNV?

        RESOLVED:  These functions minimize the state changes needed
        to depth test path rendering consisting of several co-planar
        path layers (as is typical of path rendering content) against
        conventional depth-tested 3D rendering.

        To properly depth test path rendering against conventional 3D
        rendering and other path rendering, particularly when a set of
        paths layer upon themselves, it is necessary to pull forward
        slightly the depth values generated during the stenciling step.
        This avoids Z-fighting when drawing path rendered layers that
        are logically co-planar.  However when covering pixels (assuming
        the stencil test passed during covering), we unconditionally
        write un-offset depth values.

        To depth-test path rendered content in this manner, follow the
        following pattern:

        Perform the following initialization:

            // Conventional initialization for depth testing and using path rendering
            glEnable(GL_DEPTH_TEST);
            glStencilFunc(GL_NOT_EQUAL, 0, 0xFF);
            glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);

            // The additional calls for depth testing of path-rendering...
            glPathStencilDepthOffsetNV(-0.05, -1);  // push stenciled path depth values slightly closer
            glPathCoverDepthFuncNV(GL_ALWAYS);

        Clear the framebuffer, including the depth buffer:

            // Clearing
            glClear(GL_COLOR_BUFFER_BIT |
                    GL_STENCIL_BUFFER_BIT |
                    GL_DEPTH_BUFFER_BIT);

        For each rendered path object...

            0)  Make sure stencil testing is enabled (in case it was
                disabled to draw prior conventional 3D objects).

                glEnable(GL_STENCIL_TEST);

            1)  Stencil step:

                glStencilFillPathNV(pathObj, GL_COUNT_UP_NV, 0xFF);

            2)  Cover step:

                glCoverFillPathNV(pathObj, GL_COUNT_UP_NV, 0xFF);

        For conventional 3D objects...

            0)  Make sure stencil testing is disabled (in case it was
                enabled to draw prior path rendered objects).

                glDisable(GL_STENCIL_TEST);

            1)  Draw normally:

                draw_3d_object_normally();

        With this pattern, conventional and path rendered objects can
        be rendered in arbitrary order.

        The above pattern shows path filling, but path stroking works
        the same.

        Notice only the initialization calls to
        glPathStencilDepthOffsetNV and glPathCoverDepthFuncNV are
        actually "different" than conventional 3D or path rendering.

        One potential disadvantage of this approach is that other objects
        with nearly identical depth values to the depth values of the
        path rendering content may be judged to pass the depth test when
        technically the other object's depth values are slightly closer.
        This is because the path stencil depth offset is pushing path
        rendering depth values slightly closer.  While this is possible,
        this occurs in situations where the proper occlusion was nearly
        ambiguous because the depth values between the other object and
        the path rendering are so close.

   121. What if the disadvantage of depth values having to be offset closer
        is deemed unacceptable (this will be rare).

        RESOLVED: If 100% exact depth occlusion is crucial to the
        application, this can be achieved at some cost be stenciling
        a planar conservative bounding region for the path object into
        the stencil buffer, depth testing the rendering of this plane.
        This depth-tested plane region should not perform color writes
        but should set the most-significant bit of the stencil buffer
        (for an 8-bit stencil buffer, done with GL_REPLACE of 128).
        Then path rendering, with depth testing DISABLED, can use
        the glPathStencilFunc to discard stencil values without the
        most-significant bit set.  Finally the plane region must
        be redrawn again to clear any stencil values left with the
        most-significant bit set.  This approach essentially uses the
        depth plane region as a depth-tested proxy for the proper depth
        values for the path rendering.

   122. How should the path stencil depth offset be described?

        RESOLVED:  The function is named glPathStencilDepthOffsetNV and
        the query tokens are GL_PATH_STENCIL_DEPTH_OFFSET_FACTOR and
        GL_PATH_DEPTH_OFFSET_UNITS.

        Note that "polygon offset" does not appear in the name.  Polygon
        offset isn't appropriate in the context of path rendering because
        paths aren't technically polygon primitives.  The term "depth
        offset" is the actual name of the functionality that offsets
        depth values of polygons (the name of section 3.6.4 specifying
        glPolygonOffset is actually titled "Depth Offset").

        There's no perfect attribute category of state for path stencil
        depth offset factor and units to belong so the "polygon" category
        just like the polygon offset state.

   123. Does glPointAlongPathNV have anything to do with the path's
        dashing state?

        RESOLVED:  No.

        The arc length computation necessary for glPointAlongPathNV
        computes the arc length along the path (really a subpath) but
        ignores any gaps created by the dash pattern.

        Knowing the dash count, pattern, offset, and reset state, you
        could adjust the distance passed to glPointAlongPathNV to account
        for dashing gaps, but this is something the application must do.

   124. Should PostScript user path parser enforce the same error
        conditions as PostScript?

        RESOLVED:  No.

        Section 4.6.1 (User Path Construction) in the PostScript Language
        Reference Manual explains user paths.

        The section includes some restrictions.  The "ucache" operator
        is optional but must be the first operator in a user path.
        The "setbbox" operator is required.  The next operator must be an
        "absolute positioning operator (moveto, arc, or arcn)."

        The grammar in 5.X.1.2.2 (PostScript Path Grammar) does not
        enforce these restrictions.  In particular, the operators can
        appear in any order and none are required.

        The rationale for this relaxed behavior is: 1) to make the
        parser easier to specify, 2) make the specification of a path
        object through a PostScript path grammar more consistent with
        specifying a path using glPathCommandsNV, and 3) not require
        specification of a user path bounding box that isn't relevant
        in the context of OpenGL rendering.

        If a path command is used without a prior absolute positioning
        command, the initial position is assumed to be (0,0).  So a string
        such as "40 50 lineto" would draw a line from (0,0) to (40,50).

   125. The ISO PDF 32000 standard has additional path construction
        operators for rectangles and cubic Bezier curves with duplicated
        first or last control points.  Should this extension have
        first-class path commands for these operators?

        RESOLVED:  Yes.  These PDF operators correspond
        to the path commands GL_DUP_FIRST_CUBIC_CURVE_TO_NV,
        GL_DUP_LAST_CUBIC_CURVE_TO_NV, and GL_RECT_NV, corresponding to
        the operators "v", "y", and "r" respectively.

        See Table 59 (Path Construction Operators) in the PDF 32000-1:2008
        specification (page 133).  See:

            http://www.adobe.com/devnet/acrobat/pdfs/PDF32000_2008.pdf  [[ free version ]]
            http://www.iso.org/iso/iso_catalogue/catalogue_ics/catalogue_detail_ics.htm?csnumber=51502

        These additional operators make path specification and storage
        more compact, help editing, and better semantically match the
        important PDF standard.  PDF supports just absolute versions of
        these commands so relative versions are NOT provided.

        The "m", "l'", "c", and "h" operators correspond to GL_MOVE_TO_NV,
        GL_LINE_TO_NV, GL_CUBIC_CURVE_TO_NV, and GL_CLOSE_PATH_NV
        respectively.

        There is not a string grammar for glPathStringNV to encode
        PDF commands.

   126. What is the GL_RESTART_PATH_NV path command for?

        RESOLVED:  It is useful to be able to concatenate path
        command sequences as if they are independent from each other.
        The GL_RESTART_PATH_NV provides a way to reset the state of
        path command processing back to its initial state when the first
        command of a path's command sequence is processed.

        So you could use glPathSubCommandsNV to append a path sequence
        to an existing path object's sequence.  By first appending a
        GL_RESTART_PATH_NV command, you make sure the result is consistent
        with drawing the path sequences independently.

        Specifically, /sp/, /cp/ and /pep/ are re-initialized to (0,0) when
        a GL_RESTART_PATH_NV path command is encountered.

        Additionally, this restart also has the effect of causing
        CIRCULAR_TANGENT_ARC_TO_NV to NOT draw an initial tangent line
        segment.  So if you want to draw multiple independent circular
        arcs using the GL_CIRCULAR_TANGENT_ARC_TO_NV parameterization,
        you need a GL_RESTART_PATH_NV command just prior to each
        GL_CIRCULAR_TANGENT_ARC_TO_NV command.  In this respect,
        GL_RESTART_PATH_NV is different from a GL_MOVE_TO_NV command to
        (0,0).

        The GL_RESTART_PATH_NV does not by itself reset the dash
        offset, but if the path's GL_PATH_DASH_OFFSET_RESET_NV is set
        to GL_MOVE_TO_RESETS_NV, GL_RESTART_PATH_NV (as with any command
        that updates /sp/) will reset the dash offset.

   127. <<Bogus issue removed.>>

   128. How does the "stencil" and "cover" steps operate on a multisample
        framebuffer when the GL_MULTISAMPLE enabled is disabled?

        RESOLVED:  The "stencil" step respects the disabled GL_MULTISAMPLE
        enable and rendered aliased stencil coverage.

        When MULTISAMPLE is disabled, "stencil" step coverage
        determinations are made at the pixel center.  (This will result
        in an aliased appearance for the determined path coverage so
        stenciling and covering paths with GL_MULTISAMPLE disabled isn't
        recommended.)

        All the (non-masked) stencil samples for the pixel are considered
        covered or not based on the pixel center's coverage determination.
        For example, if MULTISAMPLE is disabled for a multisample
        buffer and the pixel center is determined covered during
        glStencilFillPathNV or glStencilStrokePathNV (or instanced
        versions), all the samples are updated (INCR/DECR/INVERT for
        filling or REPLACE for stroking).

        "non-masked" means that the glSampleMaskIndexedNV state applies.

        The "cover" step also respects the disabled GL_MULTISAMPLE enable.

        To maintain rendering invariances in order to guarantee
        conservative covering, both the "stencil" and "cover" step should
        be rendered with the same GL_MULTISAMPLE enable state.

   129. What happens when a command or query takes a sequence of path object
        names and a named path object does not exist?

        RESOLVED:  The non-existent path is "skipped" in instanced
        commands such as glStencilFillPathInstancedNV (and the transform
        for the particular path name is skipped over).  Notice the
        pseudo-code for these instanced path commands uses glIsPathNV
        to test if each path name exists.

        Queries cannot simply ignore the invalid name as they return
        information.  glGetPathSpacingNV treats the non-existent name as
        having zero space.  glGetPathMetricRangeNV and GetPathMetricsNV
        return metric values of -1 for the metrics of non-existent
        path objects (as also occurs if the path object lacks metrics
        information).

        No GL error is generated due to a non-existent path name.

        (Early implementation prior to NVIDIA's OpenGL 4.3 implementation
        might crash or generate a GL_INVALID_OPERATION error.
        The behavior was a bug.)

   130. Should glCallLists be extended to take the GL_UTF8_NV and
        GL_UTF16_NV date types?

        RESOLVED: No.  That might make sense in another extension since it
        would allow complex Unicode text to be rendered by glCallLists.

        (An early version of this specification did call for supporting
        UTF sequences for glCallLists, but that behavior was never
        implemented and is now purged from the specification.)

   131. Should glPathTexGenNV and glPathColorGenNV transform the plane
        equations for GL_EYE_LINEAR by the inverse transpose modelview
        matrix?

        RESOLVED:  Yes.

        This matches the way glTexGenfv operates with GL_EYE_LINEAR
        texgen planes.  This allows the eye plane equations to be
        specified in the current object-space.

        (Early specifications, prior to revision 9, incorrectly failed
        to specify this transformation.)

   132. Should new commands and queries be used to support generating
        GLSL fragment inputs?

        RESOLVED:  Add a new command to specify the path fragment
        input generation state but use the API introduced by the
        ARB_program_interface_query extension specification to query
        back the path fragment input generation state.

        The new command is glProgramPathFragmentInputGenNV.  Given a GLSL
        program object and a GL_FRAGMENT_INPUT_NV resource location,
        this command provides the linear function state with which to
        generate the specified interpolated fragment input.

        The new GL_FRAGMENT_INPUT_NV token names the path fragment input
        resource.

        The new program resource properties GL_PATH_GEN_MODE_NV,
        GL_PATH_GEN_COEFF_NV, and GL_PATH_GEN_COMPONENTS_NV name the
        path fragment input generation resources.

   133. How should this specification interact with a Core profile context?

        RESOLVED:  See "Dependencies on Core Profile and OpenGL ES"
        section.

        In summary:

        Enough modelview and projection functionality from
        EXT_direct_state_access is required to make transformations of
        paths possible.

        Fragment varyings of GLSL programs can be interrogated and these
        can be generated.

        Exclude fixed-function fragment varying commands, queries,
        and GLSL built-in variables.

   134. Existing path rendering systems typically specify 2D transforms.
        Such transforms are cheaper to load, concatenate, and render with.
        How are 2D transforms specified?

        RESOLVED:  Add new matrix commands.

        Driver implementations can exploit these more compact matrix
        representations to accelerate path rendering where often matrix
        changes are frequent relative to the amount of rendering.  The
        concatenation of two 3x2 matrices is 24 multiply-add operations;
        while the concatenation of two 4x4 matrices is 64 multiply-add
        operations, so requiring over 2.5x more math operations.

        This table shows the correspondence between other path rendering
        APIs and the corresponding matrix routine so we need only populate
        the range of matrix representations used by major path rendering
        standards.

        Standard         Type              Component order  Corresponding GL load command
        ---------------  ----------------  ---------------  -----------------------------
        Direct2D         D2D_MATRIX_3X2_F  [0,2,4]          glMatrixLoad3x2fNV
                                           [1,3,5]
        Cairo            cairo_matrix_t    [0,2,4]          glMatrixLoad3x2fNV
                                           [1,3,5]
        Skia             SkMatrix          [0,1,2]          glMatrixLoadTranspose3x3fNV
                                           [3,4,5]
                                           [6,7,8]
                         SkScalar [6]      [0,2,4]          glMatrixLoad3x2fNV
                                           [1,3,5]
        Qt               QMatrix           [0,2,4]          glMatrixLoad3x2fNV
                                           [1,3,5]
        OpenVG           VGfloat [9]       [0,3,6]          glMatrixLoad3x3fNV
                                           [1,4,7]
                                           [2,5,8]
        AGM              BRVCoordMatrix    [0,2,4]          glMatrixLoad3x2fNV
                                           [1,3,5]
        Ghostscript      gs_matrix         [0,2,4]          glMatrixLoad3x2fNV
                                           [1,3,5]

        Along with the glMatrixLoad*NV commands, there are corresponding
        glMatrixMult*NV commands.

        Queries should be rare so the existing queries returning all 16
        values of a current matrix are sufficient.

   135. Does the "layout(location=2)", etc. syntax work for fragment inputs?

        RESOLVED:  Yes, assuming separate shader objects support.
        The ARB_separate_shader_objects functionality (made core in
        OpenGL 4.1) supports layout qualifiers to annotate locations on
        arbitrary fragment shader inputs.

        Example: A fragment shader could include the statement:

            layout(location=4) in vec4 eye_space;

        This would ensure that the location queried with
        GetProgramResourceLocation(program, GL_FRAGMENT_INPUT_NV,
        "eye_space") will return 4.

   136. What data types work with glProgramPathFragmentInputGenNV?

        RESOLVED:  Just floating-point scalars and vectors.

        Half-precision and double-precision varyings considered
        floating-point, and hence are allowed, but implementations may
        interpolate double-precision at single-precision.

        Matrix, array, structure, boolean, and integer data types are
        not supported.

        Generated values are intrinsically floating-point (they are
        basically interpolants) hence the floating-point restriction.

        Restricting the generation to floating-point scalars and vectors
        shouldn't be a hardship.

   137. How can a fragment varying (or fragment input for GLSL) be driven
        to a constant value?

        RESOLVED:  Use the GL_CONSTANT genMode for this.

        Example:  This command:

            GLfloat float4_constant[4] = { 1, 2, 3, 4 };
            glPathTexGenNV(GL_TEXTURE0, GL_CONSTANT, 4, float4_constant);

        is equivalent to:

            GLfloat coefficients[3*4] = { 0,0,1, 0,0,2, 0,0,3, 0,0,4 };
            glPathTexGenNV(GL_TEXTURE0, GL_OBJECT_LINEAR, 4, coefficients);

        In the latter form, the zeros in the coefficients array would be
        multiplied by the object-space X and Y so would always evaluate
        (s,t,r,q) to (1,2,3,4) just as the former GL_CONSTANT version
        would.

        GL_CONSTANT also works with glPathColorGenNV and
        glProgramPathFragmentInputGenNV.

   138. What happens to fragment inputs that are not configured by
        glProgramPathFragmentInputGenNV?

        RESOLVED:  Such variables are forced to constant zero.

        The default genMode is GL_NONE and this results in a fragment
        input outputting sc, tc, rc, and qc for its first, second,
        third, and fourth components respectively.  These values
        are all zero in the case of fragment input generation with
        glProgramPathFragmentInputGenNV (whereas with glPathTexGenNV,
        they take on the value of the respective texture coordinate
        set's current values).

   139. The fragment input generation state includes floating-point coefficients
        but the ARB_program_interface_query extension provides no way
        to query floating-point state so how can this state be queried?

        RESOLVED:  This extension adds glProgramResourceIndexfvNV to
        allow floating-point program resource state to be queried.

   140. Can a program object with a vertex shader be used to cover paths?

        RESOLVED:  Yes.

   141. Is there a technical explanation of this extension beyond the
        specification itself?

        RESOLVED:  Yes, check out the SIGGRAPH Asia 2012 paper
        "GPU-accelerated Path Rendering":

           https://dl.acm.org/citation.cfm?id=2366145.2366191
           http://developer.download.nvidia.com/devzone/devcenter/gamegraphics/files/opengl/gpupathrender.pdf

        There is an accompanying annex to this paper titled "Programming
        NV_path_rendering":

            http://developer.nvidia.com/sites/default/files/akamai/gamedev/files/nvpr_annex.pdf

   142. Should conic sections (rational quadratic Bezier segments) be supported?

        RESOLVED:  Yes, Skia supports these.

        The GL_CONIC_CURVE_TO_NV and GL_RELATIVE_CONIC_CURVE_TO_NV path
        commands take five path coordinates:

           x1,y1, x2,y2, w

        The first two pairs of coordinates are control points similar to
        the GL_QUADRATIC_CURVE_TO_NV and GL_RELATIVE_QUADRATIC_CURVE_TO_NV
        path commands.  The fifth coordinate "w" is a homogeneous coordinate
        that applies to the middle (extrapolating) control point.

        Skia parameterizes its SkPath::kConic_Verb conic curve path
        command in the same manner.  (See Skia's SkConic class in
        skia/include/core/SkGeometry.h for details.)

        When the "w" is 1.0, the behavior of the GL_CONIC_CURVE_TO_NV and
        GL_RELATIVE_CONIC_CURVE_TO_NV commands behave identically to the
        GL_QUADRATIC_CURVE_TO_NV and GL_RELATIVE_QUADRATIC_CURVE_TO_NV
        commands respectively; this case corresponds to a parabolic segment.

        When "w" is less than 1.0, the resulting conic is a partial
        elliptical arc.  When "w" is greater than 1.0, the resulting
        conic is a hyperbolic arc.

        See Table 5.pathEquations (Path Equations) for the specific
        rational quadratic Bezier equations for the GL_CONIC_CURVE_TO_NV
        and GL_RELATIVE_CONIC_CURVE_TO_NV path commands.

        The GL_RELATIVE_CONIC_CURVE_TO_NV path command is not supported
        by Skia but is trivial to support and maintains a symmetry that
        general-purpose path commands should have relative versions.

   143. What happens when the "w" (5th coordinate) of a conic section
        path command is non-positive?

        RESOLVED:  Match Skia's behavior and treat the path command as
        a line segment from the current control point to the interpolating
        control point (x2,y2).

        At the limit when w nears zero, partial elliptical arcs would
        become a line segment.

   144. Should "smooth" conic sections be supported similar to
        GL_SMOOTH_QUADRATIC_CURVE_TO_NV?

        RESOLVED:  No.  Conceptually, there's no problem supporting
        smooth conic sections, however no standard supports smooth conic
        sections to justify the feature.

   145. Should there be a "Character alias" for the absolute and relative
        conic curve commands?

        RESOLVED:  Yes, "W" for GL_CONIC_CURVE_TO_NV and "w" for
        GL_RELATIVE_CONIC_CURVE_TO_NV are appropriate.

   146. Should there be a rational cubic path command?

        RESOLVED:  No way!

        Rational cubic segments are subject to topological transitions
        when transformed projectively (as is possible when a path is
        transformed by the--potentially projective--modelview-projection
        transform!).

   147. Why are rounded rectangles supported?

        RESOLVED:  Rounded rectangles are popular in web content.  The W3C's
        "CSS Backgrounds and Borders Module Level 3" candidate recommendation
        specifies rounded rectangles and they are popular in web content.

            http://www.w3.org/TR/css3-background/

        Native paths commands for rounded rectangles allow such content
        to be specified and rasterized with less overhead than comparable
        specification of the same path with multiple line and arc (or
        conic) commands.

            http://www.w3.org/TR/css3-background/#corners

   148. Should multiple parameterization for rounded rectangles be
        supported?

        RESOLVED:  Yes.

        Both circular and elliptical corners are supported with either
        uniform (all corners have the same x- and y-axis radii) or
        per-corner radii.

        Also relative and absolute versions are supported (including
        adding an absolute version of GL_RECT_NV for completeness).

   149. Should negative x, y, width, height, and radii parameters be allowed
        for rectangles and rounded rectangles?

        RESOLVED:  Yes.  The formulas operate reasonably with negative values.

        Negative values allow the winding order to be reversed.

        GL_RECT_NV already allowed negative values.

   150. Should the "stencil" and "cover" path operations be combined
        into a single command?

        RESOLVED:  Yes, these commands are commonly used in sequence and
        profiling shows combining the commands can provide a small but
        measurable CPU efficiency benefit by reducing name translation,
        object locking overhead, and error checking.
        See the commands:

            glStencilThenCoverFillPathNV
            glStencilThenCoverStrokePathNV

        These commands are specified to behave like a "stencil" command
        on a path followed immediately by a "cover" command on the same
        path.

        There are also instanced versions:

            glStencilThenCoverFillPathInstancedNV
            glStencilThenCoverStrokePathInstancedNV

        These commands can be display listed.

        Indeed, one advantage of the Instanced versions of the
        glStencilThenCover* commands is the instanced array can be copied
        into the display list once.  While a display list optimizer
        could recognize this same benefit, it is simpler to be explicit
        that there is a single set of transform values used by both the
        instanced "stencil" and "cover" operations.

   151. Should there be a way to get glyph indices for a particular font face
        to perform advanced text shaping?

        RESOLVED:  Yes, see glPathGlyphIndexArrayNV,
        glPathMemoryGlyphIndexArrayNV, and glPathGlyphIndexRangeNV.

        Advanced text shaping APIs such as HarfBuzz and Pango generate
        combine text with a font face and provide a sequence of glyph
        indices with corresponding positions to render the text.

        Mozilla and Google have both confirmed the requirement for this.

        Advanced text shaping requires more knowledge of scripts and
        font metrics than can be expressed through NV_path_rendering.
        There is no interest to attempt, or even try to attempt, exposing
        sufficient font metrics for advanced text shaping.  Instead
        the presumption is that one or more higher-level libraries
        (e.g. HarfBuzz + FreeType 2) are used to perform text shaping.

        While glGetPathSpacingNV is useful and sufficient for providing
        basic kerning of Latin and other common scripts, but is
        well-understood to be insufficient for advanced text shaping.

   152. Should glPathGlyphIndexRangeNV take the range of path objects
        as a parameter or return the base & count of path names created
        from the specified font's glyph indices?

        RESOLVED:  glPathGlyphIndexRangeNV operates like glGenPathsNV
        to first get an unassigned range of path object names based on
        the number of glyph indices in the font face.  Then specifies
        the path object for every glyph index.

        This requires returning a pair of GLuint values for the base
        and count.  Additionally there is a return value to indicate
        whether and, if not why not, the path objects for the glyphs
        are assigned and specified.

   153. Why were glPathGlyphIndexArrayNV and glPathMemoryGlyphIndexArrayNV
        added?

        RESOLVED:  Web browsers such as Chrome and other applications
        relying on glyph indices rely on arranging glyph indices of
        several fonts together so controlling the order of glyph index
        arrangement proves important.  glPathGlyphIndexRangeNV returns
        a dynamically allocated range (implicitly using glGenPathsNV)
        and this proved not well-suited for actual use of glyph indices.

   154. Why glPathMemoryGlyphIndexArrayNV added?

        RESOLVED: Also web sites today very often provide server-supplied
        fonts via the Web Open Font Format (WOFF) standard.  This means
        fonts are provided by font representations in system memory
        rather than accessed by file name or system font name.

   155. Why is GL_FONT_NUM_GLYPH_INDICES_NV added?

        RESOLVED:  This is a relevant per-font parameter that is useful
        to ensure that glyphs used by glyph index know the proper bounds
        on the glyph indices.

        This per-font parameter is an integer so is not scaled by the
        emScale.

   156. Why does glPathMemoryGlyphIndexArrayNV take a face index?

        RESOLVED:  Implementations are likely to use FreeType 2's
        FT_New_Memory_Face to implement this functionality.  The first
        face index is zero so normally zero should be passed for the
        face index.

   157. If a face index for glPathMemoryGlyphIndexArrayNV corresponds to
        a bitmap font or otherwise isn't suitable for providing path
        objects, what happens?

        UNRESOLVED:  Probably GL_FONT_UNINTELLIGIBLE_NV should be
        returned.

   158. Is glyph index zero special?

        RESOLVED:  According to SNFT conventions, glyph index zero
        corresponds to the font face's missing glyph.  Therefore at
        least once glyph outline should always exist.

   159. Why as GL_FONT_CORRUPT_NV renamed to GL_FONT_UNINTELLIGIBLE_NV?

        RESOLVED:  Because it is hard to distinguish between a font being
        corrupt and simply not being supported by the implementation.
        Unintelligible is less misleading and more honest about the
        situation.

   160. Are there alternatives to STANDARD_FONT_FORMAT_NV?

        RESOLVED:  Not currently.  There might be a need in the future
        to identify fonts or glyph outlines with some other token if
        the font does use the SNFT format.  PostScript, TrueType, and
        OpenType font formats are all SNFT formats.  The Web Open Font
        Format should be supportable too because it contains a magic
        number with which to identify the format of the binary data.

   161. Is the memory provided by glPathMemoryGlyphIndexArrayNV referenced
        after the command is issued?

        RESOLVED:  No.  The GL implementation is responsible for copying
        from the system memory buffer provided.  This likely requires
        copying the entire buffer.

        (Perhaps another font target to allowed referenced access to the
        font data may be a good idea though it would likely require all
        path objects specified by glPathMemoryGlyphIndexArrayNV to be
        deleted before freeing the memory.  Referencing client system
        memory is generally considered taboo for GL implementations
        beyond the duration of a GL command or query's execution however.
        Copying the buffer avoids any ambiguity and provides for reliable
        operation, tracing, and network extensibility.)

   162. What if glPathGlyphIndexArrayNV or glPathMemoryGlyphIndexArrayNV
        attempt to specify more path objects than the font supports
        glyph indices?

        RESOLVED:  Path objects that would correspond to glyph indices
        that are beyond the maximum glyph index in the font face are
        not disturbed.

        For example, if a font face contains 258 glyph indices, but
        the numGlyphs parameter to glPathGlyphIndexArrayNV is 300, the
        command silently acts as if 258 glyph indices were requested.
        No GL error is generated in this case.  Also the path objects
        named firstPathName+258 and beyond are not disturbed.

        The rationale for this behavior is to avoid needless errors or
        complexity if an application overestimates the number of glyph
        indices a font has.

   163. What concrete reasons might GL_FONT_TARGET_UNAVAILABLE_NV be
        generated?

        RESOLVED:  Here are some situations...

        The Win32 API lacks a way to load a font from a file name.
        If FreeType 2 is unavailable (say its DLL is missing or
        the GL implementation simply does not support it), this
        would cause use of the GL_FILE_NAME_NV target to return
        GL_FONT_TARGET_UNAVAILABLE_NV.

        Linux implementations for the X Window System are likely to use
        FontConfig to map system font names (such as "Arial") to some
        font file.  If the FontConfig shared library is unavailable,
        cannot be initialized, is a very old version, or its configuration
        files are missing or corrupt, the GL_SYSTEM_FONT_NAME_NV font
        target could return GL_FONT_TARGET_UNAVAILABLE_NV.

        These situations are possible and so applications should
        anticipate that GL_FONT_TARGET_UNAVAILABLE_NV might be returned
        but properly configured systems should not be returning this
        value.  Developers debugging this condition should check
        ARB_debug_output messages for an explanation.

   164. What path glyph specification commands support which font targets?

        RESOLVED:

        The FILE_NAME_NV, SYSTEM_FONT_NAME_NV, and STANDARD_FONT_NAME_NV
        font targets are for glPathGlyphsNV and glPathGlyphRangeNV.

        The FILE_NAME_NV and SYSTEM_FONT_NAME_NV font targets are for
        glPathGlyphsNV, glPathGlyphRangeNV, glPathGlyphIndexArrayNV,
        and glPathGlyphIndexRangeNV.  STANARD_FONT_NAME_NV does not
        apply to these commands because standard font name support
        Unicode character point access to glyphs but not glyph indices.

        The STANDARD_FONT_FORMAT_NV font target is just for the
        glPathMemoryGlyphIndexArrayNV command.

   165. Why is the GL_PATH_STROKE_BOUND_NV parameter supported?

        RESOLVED:  The path's stroke approximation bound helps the
        GL implementation and an application bound the amount of
        approximation error allowed when cubic Bezier path segments or
        partial elliptical arcs are stenciled.

        Theory for offset curves indicates determining if a point is
        within a given offset of a cubic Bezier curve (the generating
        curve for the offset curve) amounts to solving a 5th degree
        polynomial equation.  That is not tractable in real-time graphics
        so some approximation of the actual offset curve is assumed.

        By comparison solve the point containment problem for a sample
        position with respect to the offset curve of a quadratic Bezier
        segment requires solving only a 3rd degree polynomial which is
        tractable for modern GPUs.  The assumption here is that stroke
        point containment with respect to quadratic Bezier segments
        and linear segments, as well as capping and join geometry, can
        be tractably solved without analytical approximation (though
        numerical issues may still limit the accuracy at the limits of
        available numerical precision).

        With this in mind, there should be some intuitive bound on
        the approximation error allowed.  The GL_PATH_STROKE_BOUND_NV
        path parameter provides such an intuitive limit expressed as a
        percentage of the path's stroke width.

   166. Should the radii passed to the GL_ROUNDED_RECT*_NV and
        GL_RELATIVE_ROUNDED_RECT*_NV support negative values?

        RESOLVED:  Yes.

        However, the x or y radii are negated if the width or height
        respectively is negative.  This behavior ensures that a rectangle
        with reversed winding can be specified (useful for cutting out
        rounded rectangular "holes" in paths) by simply flipping the
        width or height sign while leaving the radii values positive.

        The use of the /sign/ function in the specification of the /rrect/
        function enforces this behavior.

        This is important because the GL_ROUNDED_RECT_NV and
        GL_ROUNDED_RECT4_NV (and relative versions) specify a single
        circular radius per-rectangle or per-corner respectively without
        providing an x & y radii.  Without the /sign/ terms, it would
        not be possible to use the these commands and specify a reverse
        winding rounded rectangle.

        Still negative values are allowed for the radii and the formula
        should be applied as specified.  Negative radii permit "fins"
        and "crossed roundings" to be added rounded rectangles.

   167. Should CLOSE_PATH_NV count as specifying the start position
        (sp) for the purposes of determining if the PostScript path
        commands CIRCULAR_CCW_ARC_TO_NV or CIRCULAR_CW_ARC_TO_NV should
        change sp to ncp?

        RESOLVED:  No.  The PostScript semantic appears to be that
        a CLOSE_PATH_NV does not set the "current point" to valid.
        This is based on inspection of Ghostscript behavior.

        For this reason, the "other than CLOSE_PATH_NV" phrase is placed
        in the paragraph describing how when these PostScript arc commands
        change sp.

   168. What is the initial glProgramPathFragmentInputGenNV state for all
        fragment inputs?

        RESOLVED:  See the "Program Object Resource State" table.

        The GL_PATH_GEN_MODE_NV initial state for every fragment program
        resource is GL_NONE.

        The GL_PATH_GEN_COMPONENTS_NV initial state is zero for the
        number of path fragment input components.

        The sixteen PATH_GEN_COEFF_NV coefficient values are initially
        all zero.

   169. If glGetProgramResourceiv or glGetProgramResourcefvNV are used on
        fragment program resources that are not floating-point scalars
        are vectors, what happens?

        RESOLVED:  While glProgramPathFragmentInputGenNV cannot be used to
        change such program resources, their state can be queried but
        simply always returns the intial values.

        The rationale for this is that implementations already have
        to return the initial state for fragment inputs that have
        not yet been specified.  Also the ARB_program_interface_query
        specification specifies returning innocuous or invalid
        information in preference to generating errors when the query
        does not apply to the program resource.

        Always writing back some data in the absence of an error also
        makes it easier to notice buffer overflow errors since they are
        not skipped when GL errors are generated.

        Note that an error *should* be generated by
        glGetProgramResourceiv and glGetProgramResourcefvNV if
        GL_PATH_GEN_*_NV queries are performed on a programInterface
        other than GL_FRAGMENT_INPUT_NV.

   170. Should we allow fragment input generation on half and double
        precision GLSL attributes?

        RESOLVED:  No, just single-precision fragment inputs can be
        generated.

        Double-precision attributes only support flat interpolation and
        that makes no sense for paths.

        Half-precision attributes could be supported but have
        no particular advantage on NVIDIA GPUs as half-precision
        interpolation actually happens in single-precision anyway.

        glProgramPathFragmentInputGenNV generates GL_INVALID_OPERATION when
        passed a double-precision or half-precision fragment input
        (just as it does for any other inappropriate program resource
        such as a matrix).

   171. With glProgramPathFragmentInputGenNV, what fragment input values are
        generated when the component would normally be the texture
        coordinate set component for glPathTexGenNV?

        RESOLVED:  Zero.

        glProgramPathFragmentInputGenNV is specified in terms of PathTexGenNV
        but there are no fixed-function way to drive varyings.  But
        this specification language says such under-specified varyings
        will be zero: "Because there is no associated texture coordinate
        set, the sc, tc, rc, and qc values when discussing PathTexGenNV
        are always zero when generating fragment input variables."

   172. Should glProgramPathFragmentInputGenNV be able to control the
        path generation of "gl_" prefixed built-in variables?

        RESOLVED:  No.

        glProgramPathFragmentInputGenNV operates on only fragment inputs
        that are user-defined, scalar/vector (not matrices, structures,
        arrays, or opaque types such as samplers), and single-precsion.

        Built-ins such as gl_TexCoord[0], gl_Color, gl_FogFragCoord
        are generated with glPathTexGenNV, glPathColorGenNV, and
        glPathFogGenNV respectively.

   173. How does this extension interact with OpenGL ES 2 and 3?

        RESOLVED:  Same as the Core Profile in complete OpenGL.  See Issue
        133 and the "Dependencies on Core Profile and OpenGL ES" section.

   174. OpenGL ES normally does not use or require double-precision
        floating-point. Does the OpenGL ES version of this extension support
        the double-precision entry points MatrixLoaddEXT, etc.)?

        RESOLVED: Yes. This is a conscious choice, and only
        double-precision support for CPU-side operations is required. See
        https://github.com/KhronosGroup/OpenGL-Registry/pull/119#issuecomment-341121022
        for background.

Revision History

    Rev.    Date    Author     Changes
    ----  -------- ---------  --------------------------------------------
      2   08/26/11 mjk        Initial version
      3   05/31/12 mjk        Fix glPathStencilDepthOffsetNV to accept
                              a GLfloat second parameter; add _BIT to the
                              FONT_*_NV metric token names
      4   07/06/12 mjk        Issue #128
      5   07/27/12 mjk        Fix getPathName return value sense;
                              Issue #129 and #130; UTF-8 and UTF-16
                              decoding fixes.
      6   05/23/13 mjk        Fix typo in Table 5.pathEquations
      7   06/25/13 mjk        Fix token names missing _BIT_NV suffix
      8   08/01/13 mjk        Bad argument order in instanced example
      9   08/22/13 mjk        Fix GL_EYE_LINEAR behavior
      10  09/09/13 mjk        Add core profile + smaller matrix support
      11  09/10/13 mjk        Add conic segment path commands
      12  09/18/13 mjk        Add rounded rectangles, GL_RELATIVE_RECT_NV,
                              missing new matrix language, fix typos
      13  10/21/13 mjk        <fontStyle> parameter for PathGlyphsNV
                              and PathGlyphRangeNV
      14  11/05/13 dsn        Use consistent argument names
      15  11/08/13 mjk        Add StencilThenCover* commands
      16  11/11/13 mjk        Add PathGlyphIndexRange command
      17  01/07/14 mjk        Fix typos
      18  01/07/14 mjk        Add PathGlyphIndexArray and
                              PathMemoryGlyphIndexArray commands and
                              FONT_NUM_GLYPH_INDICES_NV path query;
                              renamed FONT_CORRUPT_NV to FONT_UNINTELLIGIBLE_NV
      19  02/12/14 mjk        Document GL_PATH_STROKE_BOUND_NV
      20  02/14/14 mjk        Fix rounded rectangle radii sign behavior
                              (see issue 166)
      21  02/19/14 mjk        Refashion the rrect function
      22  02/22/14 mjk        PostScript arc behavior; see issue 167
      23  03/06/14 mjk        Document FONT_NUM_GLYPH_INDICES_BIT_NV
                              interactions; STANDARD_FONT_FORMAT_NV only
                              for glPathMemoryGlyphIndexArrayNV
      24  03/18/14 mjk        Update glPathFragmentInputGenNV state
                              and query specification; issues 168-171
      25  03/19/14 mjk        Better NVpr 1.3 explanation
      26  03/20/14 mjk        Issue 172
      27  04/15/14 mjk        ES interactions same as Core Profile
      28  05/02/14 mjk        Updated status
      29  05/15/14 mjk        Matrix*Tranpose to Matrix*Transpose
      30  05/29/14 mjk        Release 1.3 driver details
      31  07/02/14 dkoch      Fix a variety of typos and inconsistencies
                              Update ES interactions
                              Fix pseudocode (float vs double, renderBoundingBox)
      32  07/24/14 mjk        Fix Equation 5.generalParametricArc typos,
                              thanks to Chris Hebert
      33  08/19/14 mjk        Add missing 1.3 additions to revisions section
      34  08/27/14 mjk        Remove bogus polygon offset issue 127; my mistake
      35  09/09/14 mjk        Intro: fix translate mode, add StencilThenCover
      36  11/01/17 Jon Leech  Add issue 174 on double-precision ES support
