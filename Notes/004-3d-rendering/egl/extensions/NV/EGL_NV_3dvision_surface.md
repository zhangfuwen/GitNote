# EGL_NV_3dvision_surface

Name 

    EGL_NV_3dvision_surface

Name Strings

    EGL_NV_3dvision_surface

Contact

    Greg Roth, NVIDIA (groth 'at' nvidia.com)

Contributors

    Swaminathan Narayanan, NVIDIA

IP Status

    NVIDIA Proprietary.

Status

    Complete

Version

    Last Modified Date: 02 December 2011
    Revision: 1

Number

    EGL Extension #46

Dependencies

    Requires EGL 1.4

    Written against the EGL 1.4 specification.

Overview

    NVIDIA 3D Vision provides stereoscopic 3d rendering without
    requiring applications to change their rendering methods. However
    there are cases where applications can benefit from adjusting 3D
    vision parameters directly to experiment with this functionality in
    applications not yet known to 3D Vision, to assist 3D Vision in
    setting parameters correctly for unusual situations, or to present
    application-specific user-accessible controls for 3D Vision
    parameters.

    This extension provides the ability to explicitly create a surface
    with 3D Vision support regardless of application detection.
 
IP Status

    NVIDIA Proprietary

New Procedures and Functions

    None

New Tokens

    Accepted as an attribute to the <attrib_list> parameter of
    CreateWindowSurface and CreatePbufferSurface

        EGL_AUTO_STEREO_NV                0x3136

Additions to Chapter 3 of the EGL 1.4 Specification (EGL Functions and
Errors)

    Additions to section 3.5.1 (Creating On-Screen Rendering Surfaces)

    Alter the end of the second to last paragraph:

        Attributes that can be specified in <attrib_list> include
        EGL_RENDER_BUFFER, EGL_VG_COLORSPACE, EGL_VG_ALPHA_FORMAT,
        and EGL_AUTO_STEREO_NV.

    Add before the last paragraph of section 3.5.1:
        
        EGL_AUTO_STEREO_NV specifies whether 3D Vision stereo
        (stereo override) should be enabled in the driver. The default
        value of EGL_AUTO_STEREO_NV is zero.

Issues

    None

Revision History

    Rev.    Date        Author      Changes
    ----  ------------- ---------   ----------------------------------------
      1   02 Dec 2011   groth       Split 3D Vision capability from previous extension.
