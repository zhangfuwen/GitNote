# QCOM_shader_framebuffer_fetch_noncoherent

Name

    QCOM_shader_framebuffer_fetch_noncoherent

Name Strings

    GL_QCOM_shader_framebuffer_fetch_noncoherent

Contact

    Jeff Leger, Qualcomm (jleger@qti.qualcomm.com)

Status

    Complete

Version

    Last Modified Date: April 21, 2017
    Author Revision: 2

Number

    OpenGL ES Extension #277

Dependencies

    OpenGL ES 2.0 is required.

    Requires one of more of the following OpenGL ES extensions:
    EXT_shader_framebuffer_fetch
    ARM_shader_framebuffer_fetch_depth_stencil
    ARM_shader_framebuffer_fetch
    
Overview

    Existing extensions such as EXT_shader_framebuffer_fetch and 
    ARM_shader_framebuffer_fetch_depth_stencil allow fragment
    shaders to read existing framebuffer color or depth/stencil data as input.
    This extension adds support for reading those same inputs with 
    relaxed coherency requirements.  This mode can avoid expensive
    per-primitive flushes of the pixel pipeline and may offer performance
    improvements in some implementations.

    When the relaxed coherency mode is enabled, reads of the framebuffer data
    by the fragment shader will guarantee defined results only if each sample
    is touched no more than once in any single rendering pass.  The command
    FramebufferFetchBarrierQCOM() is provided to indicate a boundary between
    passes.

New Procedures and Functions

    void FramebufferFetchBarrierQCOM(void);

New Tokens

    Accepted by the <cap> parameter of Disable, Enable, and IsEnabled, and by
    the <pname> parameter of GetIntegerv, GetBooleanv, GetFloatv, GetDoublev
    and GetInteger64v:

        FRAMEBUFFER_FETCH_NONCOHERENT_QCOM                     0x96A2

New Builtin Variables
    none

Changes to the OpenGL ES 2.0 Specification, Chapter 3

    Append the following new sub-section at the end of section 3.8.2, Shader
    Execution, page 197:

    Framebuffer Fetch Coherency
    
    Rendering is typically done coherently and in primitive order.  When an
    individual sample is covered by multiple primitives, rendering for that
    sample is performed sequentially in the order in which the primitives were
    submitted.  When a Fragment shader reads framebuffer data (using
    glLastFragData, gl_LastFragDepthARM, or an inout variable) the value will 
    always return the last fragment value written to the frame buffer.  This 
    coherent behavior is enabled by default, but an optional non-coherent  
    behavior can be enabled or disabled by calling Enable or Disable with the 
    symbolic constant FRAMEBUFFER_FETCH_NONCOHERENT_QCOM.  If the non-coherent 
    behavior is enabled, applications should split their rendering into a collection
    of passes, none of which touch an individual sample in the framebuffer more than
    once.  When non-coherent behavior is enabled, the results of framebuffer reads
    are undefined if the sample being read has been touched previously in the same
    pass.  The command

      void FramebufferFetchBarrierQCOM(void);

    specifies a boundary between passes.  Any command that causes the value of a
    sample to be modified using the framebuffer is considered to touch the sample,
    including clears, blended or unblended primitives, and BlitFramebuffer copies.
    Calling Disable with the symbolic constant FRAMEBUFFER_FETCH_NONCOHERENT_QCOM
    implicilty ends the current pass.

New State


    Get Value                             Type  Get Command  Minimum Value  Description        Section
    ---------                             ----  -----------  -------------  --------------     -------
    FRAMEBUFFER_FETCH_NONCOHERENT_QCOM    B     GetBooleanv  -              enables relaxed
                                                                            coherency for reads
                                                                            of framebuffer.

Revision History

    Version 1, 2017/01/06 - Initial draft.
    Version 2, 2017/04/21 - Extended to include depth/stencil reads.
