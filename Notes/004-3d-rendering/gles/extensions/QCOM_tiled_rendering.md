# QCOM_tiled_rendering

Name

    QCOM_tiled_rendering

Name Strings

    GL_QCOM_tiled_rendering

Contributors

    Colin Sharp
    Jeff Leger

Contacts

    Chuck Smith, Qualcomm (chucks 'at' qualcomm.com)
    Maurice Ribble, Qualcomm (mribble 'at' qualcomm.com)

Notice

    Copyright Qualcomm 2009.

IP Status

    Qualcomm Proprietary.

Status

    Complete.

Version

    Last Modified Date: August 20, 2009
    Revision: #1.6

Number

    OpenGL ES Extension #70

Dependencies

    OpenGL ES 1.0 or higher is required.

    This extension interacts with QCOM_write_only_rendering.

    This extension is written based on the wording of the OpenGL ES 2.0
    specification.

Overview

    In the handheld graphics space, a typical challenge is achieving efficient
    rendering performance given the different characteristics of the various
    types of graphics memory.  Some types of memory ("slow" memory) are less
    expensive but have low bandwidth, higher latency, and/or higher power
    consumption, while other types ("fast" memory) are more expensive but have
    higher bandwidth, lower latency, and/or lower power consumption.  In many
    cases, it is more efficient for a graphics processing unit (GPU) to render
    directly to fast memory, but at most common display resolutions it is not
    practical for a device to contain enough fast memory to accommodate both the
    full color and depth/stencil buffers (the frame buffer).  In some devices,
    this problem can be addressed by providing both types of memory; a large
    amount of slow memory that is sufficient to store the entire frame buffer,
    and a small, dedicated amount of fast memory that allows the GPU to render
    with optimal performance.  The challenge lies in finding a way for the GPU
    to render to fast memory when it is not large enough to contain the actual
    frame buffer.

    One approach to solving this problem is to design the GPU and/or driver
    using a tiled rendering architecture.  With this approach the render target
    is subdivided into a number of individual tiles, which are sized to fit
    within the available amount of fast memory.  Under normal operation, the
    entire scene will be rendered to each individual tile using a multi-pass
    technique, in which primitives that lie entirely outside of the tile being
    rendered are trivially discarded.  After each tile has been rendered, its
    contents are saved out to the actual frame buffer in slow memory (a process
    referred to as the "resolve").  The resolve introduces significant overhead,
    both for the CPU and the GPU.  However, even with this additional overhead,
    rendering using this method is usually more efficient than rendering
    directly to slow memory.

    This extension allows the application to specify a rectangular tile
    rendering area and have full control over the resolves for that area.  The
    information given to the driver through this API can be used to perform
    various optimizations in the driver and hardware.  One example optimization
    is being able to reduce the size or number of the resolves.  Another
    optimization might be to reduce the number of passes needed in the tiling
    approach mentioned above.  Even traditional rendering GPUs that don't use
    tiles may benefit from this extension depending on their implemention of
    certain common GPU operations.

    One typical use case could involve an application only rendering to select
    portions of the render target using this technique (which shall be referred
    to as "application tiling"), leaving all other portions of the render target
    untouched.  Therefore, in order to preserve the contents of the untouched
    portions of the render target, the application must request an EGL (or other
    context management API) configuration with a non-destructive swap. A
    destructive swap may only be used safely if the application renders to the
    entire area of the render target during each frame (otherwise the contents
    of the untouched portions of the frame buffer will be undefined).

    Additionally, care must be taken to avoid the cost of mixing rendering with
    and without application tiling within a single frame.  Rendering without
    application tiling ("normal" rendering) is most efficient when all of the
    rendering for the entire scene can be encompassed within a single resolve.
    If any portions of the scene are rendered prior to that resolve (such as via
    a prior resolve, or via application tiling), then that resolve becomes much
    more heavyweight.  When this occurs, prior to rendering each tile the fast
    memory must be populated with the existing contents of the frame buffer
    region corresponding to that tile.  This operation can double the cost of
    resolves, so it is recommended that applications avoid mixing application
    tiling and normal rendering within a single frame.  If both rendering
    methods must be used in the same frame, then the most efficient approach is
    to perform all normal rendering first, followed by rendering done with
    application tiling.  An implicit resolve will occur (if needed) at the start
    of application tiling, so any pending normal rendering operations will be
    flushed at the time application tiling is initiated.  This extension
    provides interfaces for the application to communicate to the driver whether
    or not rendering done with application tiling depends on the existing
    contents of the specified tile, and whether or not the rendered contents of
    the specified tile need to be preserved upon completion.  This mechanism can
    be used to obtain optimal performance, e.g. when the application knows that
    every pixel in a tile will be completely rendered or when the resulting
    contents of the depth/stencil buffers do not need to be preserved.

Issues

    (1)  How do Viewport and Scissor interact with this extension?

    RESOLVED:  They don't.  When application tiling is used, the viewport and
    scissor retain their existing values, relative to the render target, not the
    specified tile.  Therefore, all rendering commands issued between
    StartTilingQCOM and EndTilingQCOM will be subject to the same scissor, and
    will undergo the same viewport transformation, as normal rendering commands.

    (2)  How do Flush and Finish interact with this extension?

    RESOLVED:  When Flush or Finish is called while application tiling is
    active, the behavior will be as if EndTilingQCOM was called, except that the
    application tiling state will remain unchanged (meaning the active tile will
    not be reset).  This means that any pending rendering commands will be
    performed to the active tile, and application tiling will continue to be
    active for any following rendering commands.

    (3)  How does SwapBuffers interact with this extension?

    RESOLVED:  It doesn't.  If SwapBuffers is called while application tiling is
    active, the contents of the entire back buffer will be copied to the visible
    window, ignoring the active tile.  SwapBuffers will have no effect on the
    application tiling state.

    (4)  What happens if the render target is changed while application tiling
         is active?

    RESOLVED:  If the current render target is changed, either by binding a new
    framebuffer object or changing the write surface of the active framebuffer
    (either explicitly or by deleting the currently bound framebuffer or write
    surface), an implicit EndTilingQCOM will occur.  The active tile will be
    reset and application tiling will be deactivated.  This is necessary because
    the active tile may not be valid for the new render target.

    (5)  Should this extension provide a query mechanism for determining things
         such as tile offset, alignment, and size requirements so a developer
         can intelligently choose tile regions?

    RESOLVED:  No.  This information is very device-dependent and difficult to
    present in an easily understood manner.  Instead, this extension will let
    developers specify an arbitrary rectangular tile region and all these
    requirements, including subdividing the given tile into multiple tiles if
    necessary, will be handled by the driver and hardware.

    (6)  Should this extension allow multiple tiles?

    RESOLVED:  No.  While earlier versions of this extension allowed for this,
    after support for arbitrary tile sizes was added the benefit of multiple
    tiles became negligible.  Allowing multiple tiles complicated the API and
    made it much more difficult for traditional rendering and some tile-based
    rendering GPUs to support this extension.

    (7)  Should multiple render targets be supported?  They are not supported
         by either the OpenGL ES core specification or any existing OpenGL ES
         extensions.  Support could be added with some new bitmasks for the
         <preserveMask> parameter.  Should this be added now, or deferred for
         inclusion in any possible future MRT extension?

    RESOLVED:  Yes.  It is not difficult to add now and doing it now makes 
    supporting MRTs in the future easier.

New Procedures and Functions

    void StartTilingQCOM(uint x, uint y, uint width, uint height,
                         bitfield preserveMask);

    void EndTilingQCOM(bitfield preserveMask);

New Tokens

    Accepted by the <preserveMask> parameter of StartTilingQCOM and
    EndTilingQCOM

        GL_COLOR_BUFFER_BIT0_QCOM                     0x00000001
        GL_COLOR_BUFFER_BIT1_QCOM                     0x00000002
        GL_COLOR_BUFFER_BIT2_QCOM                     0x00000004
        GL_COLOR_BUFFER_BIT3_QCOM                     0x00000008
        GL_COLOR_BUFFER_BIT4_QCOM                     0x00000010
        GL_COLOR_BUFFER_BIT5_QCOM                     0x00000020
        GL_COLOR_BUFFER_BIT6_QCOM                     0x00000040
        GL_COLOR_BUFFER_BIT7_QCOM                     0x00000080
        GL_DEPTH_BUFFER_BIT0_QCOM                     0x00000100
        GL_DEPTH_BUFFER_BIT1_QCOM                     0x00000200
        GL_DEPTH_BUFFER_BIT2_QCOM                     0x00000400
        GL_DEPTH_BUFFER_BIT3_QCOM                     0x00000800
        GL_DEPTH_BUFFER_BIT4_QCOM                     0x00001000
        GL_DEPTH_BUFFER_BIT5_QCOM                     0x00002000
        GL_DEPTH_BUFFER_BIT6_QCOM                     0x00004000
        GL_DEPTH_BUFFER_BIT7_QCOM                     0x00008000
        GL_STENCIL_BUFFER_BIT0_QCOM                   0x00010000
        GL_STENCIL_BUFFER_BIT1_QCOM                   0x00020000
        GL_STENCIL_BUFFER_BIT2_QCOM                   0x00040000
        GL_STENCIL_BUFFER_BIT3_QCOM                   0x00080000
        GL_STENCIL_BUFFER_BIT4_QCOM                   0x00100000
        GL_STENCIL_BUFFER_BIT5_QCOM                   0x00200000
        GL_STENCIL_BUFFER_BIT6_QCOM                   0x00400000
        GL_STENCIL_BUFFER_BIT7_QCOM                   0x00800000
        GL_MULTISAMPLE_BUFFER_BIT0_QCOM               0x01000000
        GL_MULTISAMPLE_BUFFER_BIT1_QCOM               0x02000000
        GL_MULTISAMPLE_BUFFER_BIT2_QCOM               0x04000000
        GL_MULTISAMPLE_BUFFER_BIT3_QCOM               0x08000000
        GL_MULTISAMPLE_BUFFER_BIT4_QCOM               0x10000000
        GL_MULTISAMPLE_BUFFER_BIT5_QCOM               0x20000000
        GL_MULTISAMPLE_BUFFER_BIT6_QCOM               0x40000000
        GL_MULTISAMPLE_BUFFER_BIT7_QCOM               0x80000000

Additions to Chapter 2 of the OpenGL ES 2.0 Specification (OpenGL Operation)

    Add a new section "Rendering with Application Tiling" after section 2.13:

    "2.14 Rendering with Application Tiling

    The application may specify an arbitrary rectangular region (a 'tile') to
    which rendering commands should be restricted.

    The command

        void StartTilingQCOM(uint x, uint y, uint width, uint height,
                             bitfield preserveMask);

    specifies the tile described by <x>, <y>, <width>, <height>.  Until the next
    call to EndTilingQCOM, all rendering commands (including clears) will only
    update the contents of the render target defined by the extents of this
    tile.  The parameters <x> and <y> specify the screen-space origin of the
    tile, and <width> and <height> specify the screen-space width and height of
    the tile.  The tile origin is located at the lower left corner of the tile.
    If the size of the tile is too large for the fast memory on the device then
    it will be internally subdivided into multiple tiles.  The parameter
    <preserveMask> is the bitwise OR of a number of values indicating which
    buffers need to be initialized with the existing contents of the frame
    buffer region corresponding to the specified tile prior to rendering, or the
    single value NONE.  The values allowed are COLOR_BUFFER_BIT*_QCOM,
    DEPTH_BUFFER_BIT*_QCOM, STENCIL_BUFFER_BIT*_QCOM, and
    MULTISAMPLE_BUFFER_BIT*_QCOM.  These indicate the color buffer, the depth
    buffer, the stencil buffer, and a multisample buffer modifier, respectively.
    The multisample bits are different since they modify the meaning of the
    color, depth, and stencil bits if the active surface is a multisample
    surface.  If a multisample bit is set then the corresponding color, depth,
    and/or stencil bit will cause all the samples to be copied across the memory
    bus in devices that are using fast tiled memory, but if the multisample bit
    is not set then only a single resolved sample is copied across the bus.  In
    practice, not setting the multisample bit when rendering to a multisample
    buffer can greatly improve performance, but could cause small rendering
    artifacts in some multiple-pass rendering algorithms.  The 0-7 number is to
    specify which render target is being used.  If multiple render targets are
    not being used then 0 should be specified.  Any buffers specifed in
    <preserveMask> that do not exist in the current rendering state will be
    silently ignored (simlilar to the behavior of Clear).  If NONE is specified,
    then no buffers will be initialized.  For any buffers not initialized in
    this manner, the initial contents will be undefined.

    The values of <x>, <y>, <width> and <height> are silently clamped to the 
    extents of the render target.

    The command

        void EndTilingQCOM(bitfield preserveMask);

    notifies the driver that the application has completed all desired rendering
    to the tile specified by StartTilingQCOM.  This allows the driver to flush
    the contents of the specified tile to the corresponding region of the render
    target, and disables application tiling (resuming normal rendering).  The
    parameter <preserveMask> is specified using the same values as the
    equivalent argument of StartTilingQCOM, but indicates which buffers need to
    be preserved upon completion of all rendering commands issued with
    application tiling.  For any buffers not preserved in this manner, the
    resulting contents of the buffer regions corresponding to the active tile
    will be undefined.

GLX Protocol

    None.

Errors

    INVALID_OPERATION error is generated if StartTilingQCOM is called while
    WRITEONLY_RENDERING_QCOM is enabled or the current framebuffer is not
    framebuffer complete

    INVALID_OPERATION error is generated if EndTilingQCOM is called without a
    corresponding call to StartTilingQCOM

    INVALID_OPERATION error is generated if StartTilingQCOM is called after
    calling StartTilingQCOM without a corresponding call to EndTilingQCOM

    INVALID_OPERATION error is generated if Enable(WRITEONLY_RENDERING_QCOM)
    is called between StartTilingQCOM and EndTilingQCOM

New State

    None.

Sample Usage

    GLboolean renderTiledTriangle(GLuint x, GLuint y, GLuint width, GLuint height)
    {
        // set the active tile and initialize the color and depth buffers with
        // the existing contents
        glStartTilingQCOM(x, y, width, height,
                          GL_COLOR_BUFFER_BIT0_QCOM | GL_DEPTH_BUFFER_BIT0_QCOM);
      
        // draw the triangle
        glDrawArrays(GL_TRIANGLES, 0, 3);

        // finished with this tile -- preserve the color buffer
        glEndTilingQCOM(GL_COLOR_BUFFER_BIT0_QCOM);

        // return success
        return GL_TRUE;
    }

Revision History

    #09    08/20/2009    Chuck Smith     Cosmetic changes
    #08    08/19/2009    Maurice Ribble  Add support for multiple render targets
    #07    07/28/2009    Maurice Ribble  Clean up spec
                                         Remove multiple tile support
    #06    07/23/2009    Maurice Ribble  Updated overview to match latest spec
    #05    07/15/2009    Maurice Ribble  Changed from spec to subdivide tiles
                                         instead of returning out of memory
    #04    07/06/2009    Maurice Ribble  Update due to the AMD->Qualcomm move;
                                         general extension cleanup.
    #03    11/17/2008    Chuck Smith     Clarified the results of EndTilingQCOM
                                         for unpreserved buffers.
    #02    11/10/2008    Chuck Smith     Updates to clarify behavior; additions
                                         to the Issues section.
    #01    11/04/2008    Chuck Smith     First draft.
