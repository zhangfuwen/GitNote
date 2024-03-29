# QCOM_binning_control

Name

    QCOM_binning_control

Name Strings

    GL_QCOM_binning_control

Contributors

    Maurice Ribble

Contact

    Maurice Ribble (mribble 'at' qualcomm.com)

Notice

    Copyright Qualcomm 2010

IP Status

    Qualcomm Proprietary.

Status

    Draft

Version

    Date: May 2, 2012

Number

    OpenGL ES Extension #119

Dependencies

    OpenGL ES 1.0 is required.

    This extension is written against the OpenGL ES 2.0 specification.

Overview

    This extension adds some new hints to give more control to application
    developers over the driver's binning algorithm.

    Only change this state right before changing rendertargets or right after
    a swap or there will be a large performance penalty.

Issues

    None

New Procedures and Functions

    None

New Tokens

    Accepted by the <target> parameter of Hint:

    BINNING_CONTROL_HINT_QCOM           0x8FB0

    Accepted by the <hint> parameter of Hint:
    CPU_OPTIMIZED_QCOM                  0x8FB1
    GPU_OPTIMIZED_QCOM                  0x8FB2
    RENDER_DIRECT_TO_FRAMEBUFFER_QCOM   0x8FB3
    DONT_CARE                           0x1100

Additions to Section 5.1 (Hints) of the OpenGL ES 2.0 Specification

    Replace "target is a symoblic...." with:
    
    target is a symbolic constant indicating the behavior to be controlled, and
    hint is a symbolic constant indicating what type of behavior is desired. 
    target can be GENERATE_MIPMAP_HINT or BINNING_CONTROL_HINT_QCOM.
    GENERATE_MIPMAP_HINT  indicates the desired quality and performance of
    mipmap level generation with GenerateMipmap. When target is
    GENERATE_MIPMAP_HINT hint must be one of FASTEST, indicating that the most
    efficient option should be chosen; NICEST, indicating that the highest
    quality option should be chosen; and DONT_CARE, indicating no preference in
    the matter.  A target of BINNING_CONTROL_HINT_QCOM gives hints at what 
    binning algorithm is to be used.  When the target is BINNING_CONTROL_QCOM 
    the hint must be one of the values below:

    CPU_OPTIMIZED_QCOM                - binning algorithm focuses on lower CPU
                                        utilization (this path increases vertex
                                        processing)
    GPU_OPTIMIZED_QCOM                - binning algorithm focuses on lower GPU
                                        utilization (this path increases CPU
                                        usage)
    RENDER_DIRECT_TO_FRAMEBUFFER_QCOM - render directly to the final 
                                        framebuffer and bypass tile memory 
                                        (this path has a low CPU usage, but
                                        in some cases uses more memory 
                                        bandwidth)
    DONT_CARE                         - the driver picks which binning 
                                        algorithm to use (default)

    The Qualcomm Adreno 200 family does not support 
    RENDER_DIRECT_TO_FRAMEBUFFER_QCOM option and this hint will be ignored on that
    hardware.

    When BINNING_CONTROL_QCOM do so right before changing rendertargets or right after
    swap or there will be a large performance penalty.
    
New State

Get Value                                 Type          Command      Value
---------                                 ----          -------     -------
BINNING_CONTROL_HINT_QCOM                special      GetIntegerv   DONT_CARE

Revision History

    6/15/2010  Created.
    6/22/2010  Name changes, cleanup, add token numbers.
    10/11/2010 Simplified extension to remove some of the extra modes.
    1/27/2012  Added in RENDER_DIRECT_TO_FRAMEBUFFER_QCOM
    5/2/2012   Added IP Status
