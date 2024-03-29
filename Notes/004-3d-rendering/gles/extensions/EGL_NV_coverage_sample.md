# NV_coverage_sample

Name

    NV_coverage_sample

Name Strings

    GL_NV_coverage_sample
    EGL_NV_coverage_sample

Contact

    Gary King, NVIDIA Corporation (gking 'at' nvidia.com)

Notice

    Copyright NVIDIA Corporation, 2005 - 2007

Status

    NVIDIA Proprietary

Version

    Last Modified Date:  2007/03/20
    NVIDIA Revision: 1.0

Number

    EGL Extension #17
    OpenGL ES Extension #72

Dependencies

    Written based on the wording of the OpenGL 2.0 specification
    and the EXT_framebuffer_object specification.

    Written based on the wording of the EGL 1.2 specification.

    Requires OpenGL-ES 2.0 and OES_framebuffer_object.

    Requires EGL 1.1.

Overview

    Anti-aliasing is a critical component for delivering high-quality
    OpenGL rendering.  Traditionally, OpenGL implementations have
    implemented two anti-aliasing algorithms: edge anti-aliasing
    and multisampling.

    Edge anti-aliasing computes fractional fragment coverage for all
    primitives in a rendered frame, and blends edges of abutting
    and/or overlapping primitives to produce smooth results.  The
    image quality produced by this approach is exceptionally high;
    however, applications are render their geometry perfectly ordered
    back-to-front in order to avoid artifacts such as bleed-through.
    Given the algorithmic complexity and performance cost of performing
    exact geometric sorts, edge anti-aliasing has been used very
    sparingly, and almost never in interactive games.

    Multisampling, on the other hand, computes and stores subpixel
    (a.k.a. "sample") coverage for rasterized fragments, and replicates
    all post-alpha test operations (e.g., depth test, stencil test,
    alpha blend) for each sample.  After the entire scene is rendered,
    the samples are filtered to compute the final anti-aliased image.
    Because the post-alpha test operations are replicated for each sample,
    all of the bleed-through and ordering artifacts that could occur with
    edge anti-aliasing are avoided completely; however, since each sample
    must be computed and stored separately, anti-aliasing quality is
    limited by framebuffer storage and rendering performance.

    This extension introduces a new anti-aliasing algorithm to OpenGL,
    which dramatically improves multisampling quality without
    adversely affecting multisampling's robustness or significantly
    increasing the storage required, coverage sampling.

    Coverage sampling adds an additional high-precision geometric
    coverage buffer to the framebuffer, which is used to produce
    high-quality filtered results (with or without the presence of a
    multisample buffer).  This coverage information is computed and stored
    during rasterization; since applications may render objects where the
    specified geometry does not correspond to the visual result (examples
    include alpha-testing for "imposters," or extruded volume rendering
    for stencil shadow volumes), coverage buffer updates may be masked
    by the application, analagous to masking the depth buffer.
    
IP Status

    NVIDIA Proprietary

New Procedures and Functions

    void CoverageMaskNV( boolean mask )
    void CoverageOperationNV( enum operation )

New Tokens


    Accepted by the <attrib_list> parameter of eglChooseConfig
    and eglCreatePbufferSurface, and by the <attribute>
    parameter of eglGetConfigAttrib

    EGL_COVERAGE_BUFFERS_NV           0x30E0
    EGL_COVERAGE_SAMPLES_NV           0x30E1

    Accepted by the <internalformat> parameter of
    RenderbufferStorageEXT and the <format> parameter of ReadPixels

    COVERAGE_COMPONENT_NV             0x8ED0

    Accepted by the <internalformat> parameter of
    RenderbufferStorageEXT

    COVERAGE_COMPONENT4_NV            0x8ED1

    Accepted by the <operation> parameter of CoverageOperationNV

    COVERAGE_ALL_FRAGMENTS_NV         0x8ED5
    COVERAGE_EDGE_FRAGMENTS_NV        0x8ED6
    COVERAGE_AUTOMATIC_NV             0x8ED7

    Accepted by the <attachment> parameter of
    FramebufferRenderbuffer, and GetFramebufferAttachmentParameteriv

    COVERAGE_ATTACHMENT_NV            0x8ED2

    Accepted by the <buf> parameter of Clear

    COVERAGE_BUFFER_BIT_NV            0x8000

    Accepted by the <pname> parameter of GetIntegerv

    COVERAGE_BUFFERS_NV               0x8ED3
    COVERAGE_SAMPLES_NV               0x8ED4

Changes to Chapter 4 of the OpenGL 2.0 Specification

    Insert a new section, after Section 3.2.1 (Multisampling)

    "3.2.2 Coverage Sampling

    Coverage sampling is a mechanism to antialias all GL primitives: points,
    lines, polygons, bitmaps and images.  The technique is similar to
    multisampling, with all primitives being sampled multiple times at each
    pixel, and a sample resolve applied to compute the color values stored
    in the framebuffer's color buffers.  As with multisampling, coverage
    sampling resolves color sample and coverage values to a single, displayable
    color each time a pixel is updated, so antialiasing appears to be automatic
    at the application level.  Coverage sampling may be used simultaneously 
    with multisampling; however, this is not required.

    An additional buffer, called the coverage buffer, is added to
    the framebuffer.  This buffer stores additional coverage information
    that may be used to produce higher-quality antialiasing than what is
    provided by conventional multisampling.

    When the framebuffer includes a multisample buffer (3.5.6), the
    samples contain this coverage information, and the framebuffer
    does not include the coverage buffer.

    If the value of COVERAGE_BUFFERS_NV is one, the rasterization of
    all primitives is changed, and is referred to as coverage sample
    rasterization.  Otherwise, primitive rasterization is referred to
    as multisample rasterization (if SAMPLE_BUFFERS is one) or
    single-sample rasterization (otherwise).  The value of
    COVERAGE_BUFFERS_NV is queried by calling GetIntegerv with <pname>
    set to COVERAGE_BUFFERS_NV.

    During coverage sample rasterization the pixel fragment contents
    are modified to include COVERAGE_SAMPLES_NV coverage values.  The
    value of COVERAGE_SAMPLES_NV is an implementation-dependent
    constant, and is queried by calling GetIntegerv with <pname> set
    to COVERAGE_SAMPLES_NV.

    The command

      CoverageOperationNV(enum operation)

    may be used to modify the manner in which coverage sampling is
    performed for all primitives.  If <operation> is
    COVERAGE_ALL_FRAGMENTS_NV, coverage sampling will be performed and the
    coverage buffer updated for all fragments generated during rasterization.
    If <operation> is COVERAGE_EDGE_FRAGMENTS_NV, coverage sampling will
    only be performed for fragments generated at the edge of the
    primitive (by only updating fragments at the edges of primitives,
    applications may get better visual results when rendering partially
    transparent objects).  If <operation> is COVERAGE_AUTOMATIC_NV,
    the GL will automatically select the appropriate coverage operation,
    dependent on the GL blend mode and the use of gl_LastFragColor / 
    gl_LastFragData in the bound fragment program.  If blending is enabled,
    or gl_LastFragColor / gl_LastFragData appears in the bound fragment
    program, COVERAGE_AUTOMATIC_NV will behave identically to
    COVERAGE_EDGE_FRAGMENTS_NV; otherwise, COVERAGE_AUTOMATIC_NV will behave
    identically to COVERAGE_ALL_FRAGMENTS_NV.  The default coverage operation
    is COVERAGE_AUTOMATIC_NV."

    Insert a new section, after Section 3.3.3 (Point Multisample
    Rasterization)

    "3.3.4  Point Coverage Sample Rasterization

    If the value of COVERAGE_BUFFERS_NV is one, then points are
    rasterized using the following algorithm, regardless of whether
    point antialiasing (POINT_SMOOTH) is enabled or disabled.  Point
    rasterization produces fragments using the same algorithm described
    in section 3.3.3; however, sample points are divided into SAMPLES
    multisample points and COVERAGE_SAMPLES_NV coverage sample points.

    Rasterization for multisample points uses the algorithm described
    in section 3.3.3.  Rasterization for coverage sample points uses
    implementation-dependent algorithms, ultimately storing the results
    in the coverage buffer."

    Insert a new section, after Section 3.4.4 (Line Multisample
    Rasterization)

    "3.4.5  Line Coverage Sample Rasterization

    If the value of COVERAGE_BUFFERS_NV is one, then lines are
    rasterized using the following algorithm, regardless of whether
    line antialiasing (LINE_SMOOTH) is enabled or disabled.  Line
    rasterization produces fragments using the same algorithm described
    in section 3.4.4; however, sample points are divided into SAMPLES 
    multisample points and COVERAGE_SAMPLES_NV coverage sample points.

    Rasterization for multisample points uses the algorithm described in
    section 3.4.4.  Rasterization for coverage sample points uses
    implementation-dependent algorithms, ultimately storing results in
    the coverage buffer."    

    Insert a new section, after Section 3.5.6 (Polygon Multisample
    Rasterization)

    "3.5.7  Polygon Coverage Sample Rasterization

    If the value of COVERAGE_BUFFERS_NV is one, then polygons are
    rasterized using the following algorithm, regardless of whether
    polygon antialiasing (POLYGON_SMOOTH) is enabled or disabled.  Polygon
    rasterization produces fragments using the same algorithm described in
    section 3.5.6; however, sample points are divided into SAMPLES multisample
    points and COVERAGE_SAMPLES_NV coverage sample points.

    Rasterization for multisample points uses the algorithm described in
    section 3.5.7.  Rasterization for coverage sample points uses
    implementation-dependent algorithms, ultimately storing results in the
    coverage buffer."

    Insert a new section, after Section 3.6.6 (Pixel Rectangle Multisample
    Rasterization)

    "3.6.7  Pixel Rectangle Coverage Sample Rasterization

    If the value of COVERAGE_BUFFERS_NV is one, then pixel rectangles are
    rasterized using the algorithm described in section 3.6.6."

    Modify the first sentence of the second-to-last paragraph of section
    3.7 (Bitmaps) to read:

    "Bitmap Multisample and Coverage Sample Rasterization

    If MULTISAMPLE is enabled, and the value of SAMPLE_BUFFERS is one;
    or if the value of COVERAGE_BUFFERS_NV is one, then bitmaps are
    rasterized using the following algorithm. [...]"

    Insert after the first paragraph of Section 4.2.2 (Fine Control of
    Buffer Updates):

    "The coverage buffer can be enabled or disabled for writing coverage
    sample values using

        void CoverageMaskNV( boolean mask );

    If <mask> is non-zero, the coverage buffer is enabled for writing;
    otherwise, it is disabled.  In the initial state, the coverage
    buffer is enabled for writing."

    And change the text of the last 2 paragraphs of Section 4.2.2 to read:

    "The state required for the various masking operations is three
    integers and two bits: an integer for color indices, an integer for
    the front and back stencil values, a bit for depth values, and a
    bit for coverage sample values.  A set of four bits is also required
    indicating which components of an RGBA value should be written.  In the
    initial state, the integer masks are all ones, as are the bits
    controlling the depth value, coverage sample value and RGBA component
    writing.

    Fine Control of Multisample Buffer Updates

    When the value of SAMPLE_BUFFERS is one, ColorMask, DepthMask, 
    CoverageMask, and StencilMask or StencilMaskSeparate control the
    modification of values in the multisample buffer. [...]"
    
    Change paragraph 2 of Section 4.2.3 (Clearing the Buffers) to read:

    "is the bitwise OR of a number of values indicating which buffers are to
    be cleared.  The values are COLOR_BUFFER_BIT, DEPTH_BUFFER_BIT,
    STENCIL_BUFFER_BIT, ACCUM_BUFFER_BIT and COVERAGE_BUFFER_BIT_NV, indicating
    the buffers currently enabled for color writing, the depth buffer,
    the stencil buffer, the accumulation buffer and the virtual-coverage
    buffer, respectively. [...]"

    Insert a new paragraph after paragraph 4 of Section 4.3.2 (Reading Pixels)
    (beginning with "If there is a multisample buffer ..."):

    "If the <format> is COVERAGE_COMPONENT_NV, then values are taken from the
    coverage buffer; again, if there is no coverage buffer, the error
    INVALID_OPERATION occurs.  When <format> is COVERAGE_COMPONENT_NV,
    <type> must be GL_UNSIGNED_BYTE.  Any other value for <type> will
    generate the error INVALID_ENUM.  If there is a multisample buffer, the 
    values are undefined."



Modifications to the OES_framebuffer_object specification

    Add a new table at the end of Section 4.4.2.1 (Renderbuffer Objects)

    "+-------------------------+-----------------------+-----------+
     |  Sized internal format  | Base Internal Format  | C Samples |
     +-------------------------+-----------------------+-----------+
     | COVERAGE_COMPONENT4_NV  | COVERAGE_COMPONENT_NV |     4     |
     +-------------------------+-----------------------+-----------+
     Table 1.ooo Desired component resolution for each sized internal
     format that can be used only with renderbuffers"

    Add to the bullet list in Section 4.4.4 (Framebuffer Completeness)

    "An internal format is 'coverage-renderable' if it is COVERAGE_COMPONENT_NV
    or one of the COVERAGE_COMPONENT_NV formats from table 1.ooo.  No other
    formats are coverage-renderable"

    Add to the bullet list in Section 4.4.4.1 (Framebuffer Attachment
    Completeness)

    "If <attachment> is COVERAGE_ATTACHMENT_NV, then <image> must have a
    coverage-renderable internal format."

    Add a paragraph at the end of Section 4.4.4.2 (Framebuffer Completeness)

    "The values of COVERAGE_BUFFERS_NV and COVERAGE_SAMPLES_NV are derived from
    the attachments of the currently bound framebuffer object.  If the current
    FRAMEBUFFER_BINDING_OES is not 'framebuffer-complete', then both
    COVERAGE_BUFFERS_NV and COVERAGE_SAMPLES_NV are undefined.  Otherwise,
    COVERAGE_SAMPLES_NV is equal to the number of coverage samples for the
    image attached to COVERAGE_ATTACHMENT_NV, or zero if COVERAGE_ATTACHMENT_NV
    is zero."

Additions to the EGL 1.2 Specification

    Add to Table 3.1 (EGLConfig attributes)
    +---------------------------+---------+-----------------------------------+
    |        Attribute          |   Type  | Notes                             |
    +---------------------------+---------+-----------------------------------+
    |  EGL_COVERAGE_BUFFERS_NV  | integer | number of coverage buffers        |
    |  EGL_COVERAGE_SAMPLES_NV  | integer | number of coverage samples per    |
    |                           |         |    pixel                          |
    +---------------------------+---------+-----------------------------------+

    Modify the first sentence of the last paragraph of the "Buffer 
    Descriptions and Attributes" subsection of Section 3.4 (Configuration 
    Management), p. 16

    "There are no single-sample depth, stencil or coverage buffers for a
    multisample EGLConfig; the only depth, stencil and coverage buffers are 
    those in the multisample buffer. [...]"

    And add the following text at the end of that paragraph:

    "The <coverage buffer> is used only by OpenGL ES.  It contains primitive
    coverage information that is used to produce a high-quality anti-aliased
    image.  The format of the coverage buffer is not specified, and its 
    contents are not directly accessible.  Only the existence of the coverage 
    buffer, and the number of coverage samples it contains, are exposed by EGL.

    EGL_COVERAGE_BUFFERS_NV indicates the number of coverage buffers, which 
    must be zero or one.  EGL_COVERAGE_SAMPLES_NV gives the number of coverage
    samples per pixel; if EGL_COVERAGE_BUFFERS_NV is zero, then
    EGL_COVERAGE_SAMPLES_NV will also be zero."

    Add to Table 3.4 (Default values and match criteria for EGLConfig 
    attributes)

    +---------------------------+-----------+-------------+---------+---------+
    |        Attribute          |  Default  |  Selection  |  Sort   |  Sort   |
    |                           |           |  Criteria   |  Order  | Priority|
    +---------------------------+-----------+-------------+---------+---------+
    |  EGL_COVERAGE_BUFFERS_NV  |     0     |   At Least  | Smaller |    7    |
    |  EGL_COVERAGE_SAMPLES_NV  |     0     |   At Least  | Smaller |    8    |
    +---------------------------+-----------+-------------+---------+---------+
      And renumber existing sort priorities 7-11 as 9-13.

    Modify the list in "Sorting of EGLConfigs" (Section 3.4.1, pg 20)

    " [...]
      5.  Smaller EGL_SAMPLE_BUFFERS
      6.  Smaller EGL_SAMPLES
      7.  Smaller EGL_COVERAGE_BUFFERS_NV
      8.  Smaller EGL_COVERAGE_SAMPLES_NV
      9.  Smaller EGL_DEPTH_SIZE
      10. Smaller EGL_STENCIL_SIZE
      11. Smaller EGL_ALPHA_MASK_SIZE
      12. Special: [...]
      13. Smaller EGL_CONFIG_ID [...]"

Usage Examples

   (1)  Basic Coverage Sample Rasterization

        glCoverageMaskNV(GL_TRUE);
        glDepthMask(GL_TRUE);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

        while (1) 
        {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | 
                    GL_COVERAGE_BUFFER_BIT_NV);
            glDrawElements(...);
            eglSwapBuffers(...);
        }
        
   (2)  Multi-Pass Rendering Algorithms

        while (1)
        {
            glDepthMask(GL_TRUE);
            glCoverageMaskNV(GL_TRUE);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | 
                    GL_COVERAGE_BUFFER_BIT_NV);

            //  first render pass: render Z-only (occlusion surface), with 
            //  coverage info.  color writes are disabled

            glCoverageMaskNV(GL_TRUE);
            glDepthMask(GL_TRUE);
            glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
            glDepthFunc(GL_LESS);
            glDrawElements(...);

            //  second render pass: set Z test to Z-equals, disable Z-writes & 
            //  coverage writes.  enable color writes.  coverage may be 
            //  disabled, because subsequent rendering passes are rendering 
            //  identical geometry -- since the final coverage buffer will be 
            //  unchanged, we can disable coverage writes as an optimization.

            glCoverageMaskNV(GL_FALSE);
            glDepthMask(GL_FALSE);
            glDepthFunc(GL_EQUAL);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            glDrawElements(...);

            eglSwapBuffers();
        }

   (3)  Rendering Translucent Objects on Top of Opaque Objects

        while (1)
        {
            glDepthMask(GL_TRUE);
            glCoverageMaskNV(GL_TRUE);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | 
                    GL_COVERAGE_BUFFER_BIT_NV);

            // render opaque, Z-buffered geometry with coverage info for the
            // entire primitive.  Overwrite coverage data for all fragments, so
            // that interior fragments do not get resolved incorrectly.

            glDepthFunc(GL_LESS);
            glCoverageOperationNV(GL_COVERAGE_ALL_FRAGMENTS_NV);
            glDrawElements(...);

            // render translucent, Z-buffered geometry.  to ensure that visible
            // edges of opaque geometry remain anti-aliased, change the 
            // coverage operation to just edge fragments.  this will maintain 
            // the coverage information underneath the translucent geometry, 
            // except at translucent edges.

            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glCoverageOperationNV(GL_COVERAGE_EDGE_FRAGMENTS_NV);
            glEnable(GL_BLEND);
            glDrawElements(...);
            glDisable(GL_BLEND);

            eglSwapBuffers();
        }

   (4)  Rendering Opacity-Mapped Particle Systems & HUDs on Top of Opaque 
        Geometry

        while (1)
        {
            glDepthMask(GL_TRUE);
            glCoverageMaskNV(GL_TRUE);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | 
                    GL_COVERAGE_BUFFER_BIT_NV);

            // render opaque, Z-buffered geometry, with coverage info.
            glDepthFunc(GL_LESS);
            glDrawElements(...);

            // render opacity-mapped geometry.  disable Z writes, enable alpha
            // blending. also, disable coverage writes -- the edges of the 
            // geometry used for the HUD/particle system have alpha values 
            // tapering to zero, so edge coverage is uninteresting, and 
            // interior coverage should still refer to the underlying opaque 
            // geometry, so that opaque edges visible through the translucent
            // regions remain anti-aliased.

            glCoverageMaskNV(GL_FALSE);
            glDepthMask(GL_FALSE);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glEnable(GL_BLEND);
            glDrawElements(...);
            glDisable(GL_BLEND);
            
            eglSwapBuffers();
        }


Issues

    1.  Is any specific discussion of coverage sampling resolves required,
        particularly with respect to application-provided framebuffer objects?

        RESOLVED:  No.  Because the coverage sampling resolve is an
        implementation-dependent algorithm, it is always legal behavior for
        framebuffer read / copy functions to return the value in the selected
        ReadBuffer as if COVERAGE_BUFFERS_NV was zero.  This allows
        textures attached to the color attachment points of framebuffer objects
        to behave predictably, even when COVERAGE_BUFFERS_NV is one.

        Implementations are encouraged, whenever possible, to use the highest-
        quality coverage sample resolve supported for calls to eglSwapBuffers,
        eglCopyBuffers, ReadPixels, CopyPixels and CopyTex{Sub}Image.
        
    2.  Should all render buffer & texture types be legal sources for image
        resolves and coverage attachment?

        RESOLVED: This spec should not place any arbitrary limits on usage;
        however, there are many reasons why implementers may not wish to 
        support coverage sampling for all surface types.

        Implementations may return FRAMEBUFFER_UNSUPPORTED_OES from
        CheckFramebufferStatusOES if an object bound to COVERAGE_ATTACHMENT_NV
        is incompatible with one or more objects bound to DEPTH_ATTACHMENT_OES,
        STENCIL_ATTACHMENT_OES, or COLOR_ATTACHMENTi_OES.

Revision History

#1.0 - 20.03.2007

   Renumbered enumerants.  Reformatted to 80 columns.
