# NV_3dvision_settings

Name

    NV_3dvision_settings

Name Strings

    GL_NV_3dvision_settings

Contributors

    Mathias Heyer, NVIDIA
    Swaminathan Narayanan, NVIDIA
    Greg Roth, NVIDIA

Contact

    Greg Roth, NVIDIA (groth 'at' nvidia.com)

Status

    Complete

Version

    Last Modified Date: 05 Oct 2012
    Revision: 5

Number

    OpenGL ES Extension #129

Dependencies

    Written based on the wording of the OpenGL ES 2.0.25 specification

    OpenGL ES 2.0 is required.

    EGL_NV_3dvision_surface is used to specify a surface
    with support for these features explicitly enabled.

Overview

    NVIDIA 3D Vision provides stereoscopic 3d rendering without
    requiring applications to change their rendering methods. However
    there are cases where applications can benefit from adjusting 3D
    vision parameters directly to experiment with this functionality in
    applications not yet known to 3D Vision, to assist 3D Vision in
    setting parameters correctly for unusual situations, or to present
    application-specific user-accessible controls for 3D Vision
    parameters.

    This extension provides the ability to query the current stereo
    settings and adjust them as necessary.

IP Status

    NVIDIA Proprietary

New Procedures and Functions

    void StereoParameterfNV(enum pname, float param);
    void StereoParameteriNV(enum pname, int param);

New Tokens

    Accepted by the <value> parameter of GetBooleanv, GetIntegerv,
    and GetFloatv, and the <cap> parameter of isEnabled, Enable, and
    Disable:

        3DVISION_STEREO_NV                   0x90F4

    Accepted by the <value> parameter of GetBooleanv, GetIntegerv,
    and GetFloatv and the <param> parameter of StereoParameterfNV,
    and StereoParameteriNV:

        STEREO_SEPARATION_NV                 0x90F5
        STEREO_CONVERGENCE_NV                0x90F6
        STEREO_CUTOFF_NV                     0x90F7
        STEREO_PROJECTION_NV                 0x90F8

    Accepted by the <param> parameter of StereoParameterfNV and
    StereoParameteriNV and returned by GetBooleanv, GetIntegerv, and
    GetFloatv when <value> is STEREO_PROJECTION_NV.

        STEREO_PROJECTION_PERSPECTIVE_NV     0x90F9
        STEREO_PROJECTION_ORTHO_NV           0x90FA

Changes to Chapter 4 of the OpenGL ES 2.0.25 Specification (Per-Fragment
Operations and the Frame Buffer)
  
    Add to section 4.2.1, "Selecting a Buffer for Writing":

    If the bound draw surface is auto stereo enabled and 3d Vision
    rendering is enabled in the GL, color values are written to left and
    right buffers with adjustments according to current separation and
    convergence values.

    Add subsection 4.2.x, "Stereo Parameters":

    3D Vision stereo capability may be enabled at surface creation. If
    the bound draw surface is auto stereo capable, 3D Vision rendering
    can be enabled and disabled by calling Enable or Disable with the
    symbolic constant 3DVISION_STEREO_NV. If the current draw surface is
    not auto stereo capable, enabling 3D Vision rendering has no
    effect. The current state of 3D Vision rendering may be queried
    using isEnabled with <cap> set to 3DVISION_STEREO_NV.

    Various parameters control how 3D Vision stereo rendering is
    performed. Each parameter is set by calling

        void StereoParameter{if}NV(enum pname, T param);

    <pname> is a symbolic constant indicating the parameter to be set.

    If <pname> is STEREO_PROJECTION_NV, then <param> specifies the
    method of rendering 3D Vision will use. <param> must be
    STEREO_PROJECTION_PERSPECTIVE_NV or STEREO_PROJECTION_ORTHO_NV
    identifying the projection type used. If <param> has any other
    value, an INVALID_ENUM error is generated. The current 3D Vision
    stereo method is queried by calling GetIntegerv with <pname> set to
    STEREO_PROJECTION_NV. The initial value of STEREO_PROJECTION_NV is
    STEREO_PROJECTION_PERSPECTIVE_NV, but may differ for contexts
    created with a 3D Vision profile in effect.

    If <pname> is STEREO_SEPARATION_NV, then <param> specifies the
    separation value used in 3D Vision stereo rendering. The current
    separation value is queried by calling GetFloatv with <pname> set
    to STEREO_SEPARATION_NV. The initial value of STEREO_PROJECTION_NV
    depends on 3D Vision system settings at the time the context was
    created. The separation value indicates a percentage. If <param> is
    less than zero or greater than 100, an INVALID_VALUE error is
    generated.

    If <pname> is STEREO_CONVERGENCE_NV, then <param> specifies the
    convergence value used in 3D Vision stereo rendering. The current
    convergence value is queried by calling GetFloatv with <pname> set
    to STEREO_CONVERGENCE_NV. The initial value of STEREO_CONVERGENCE-
    _NV is zero, but may differ for contexts created with a 3D Vision
    profile in effect.

    If <pname> is STEREO_CUTOFF_NV,  then <param> specifies the
    separation cutoff value used in 3D Vision stereo rendering. The
    current separation cutoff value is queried by calling GetFloatv
    with <pname> set to STEREO_CUTOFF_NV. The initial value of
    STEREO_CUTOFF_NV is zero, but may differ for contexts created with a
    3D Vision profile in effect.

    Otherwise, INVALID_ENUM is generated.

    Setting stereo parameters on a context that does not support 3D
    Vision rendering has no effect on rendering.

    The state required to implement these stereo parameters is three
    floats and one integer.

Errors

    The error INVALID_ENUM is generated if StereoParameter{if} is
    called where <pname> is an unsupported enum value.

    The error INVALID_ENUM is generated if StereoParameter{if} is
    called where <pname> is STEREO_PROJECTION_NV and <param> is an
    unsupported enum value.

New State

    Add new table "Stereo State"
    Get Value                     Type    Get Command  Initial Value        Description                Sec
    ---------                     ----    -----------  -------------        -----------                ------
    3DVISION_STEREO_NV            B       isEnabled    see 4.2.x            3D Vision stereo enable     4.2.x
    STEREO_SEPARATION_NV          R       GetFloatv    see 4.2.x            3D Vision separation        4.2.x
    STEREO_CONVERGENCE_NV         R       GetFloatv    see 4.2.x            3D Vision convergence       4.2.x
    STEREO_CUTOFF_NV              R       GetFloatv    see 4.2.x            3D Vision separation cutoff 4.2.x
    STEREO_PROJECTION_NV          Z       GetIntegerv  see 4.2.x            3D Vision projection method 4.2.x

Issues

    1. What are the default values of 3D Vision stereo parameters?

    RESOLVED: The default values depend on the application being run and
    current system 3D Vision settings. For all but separation, defaults
    for an application with no 3D Vision profile. GL contexts created
    for Applications with established stereo profiles will have
    different default values for stereo parameters.

    2. Should method settings be included?

    RESOLVED: Yes, but in a form that makes sense to the user. Internal
    method settings have many forms, but all the user needs be concerned
    with are orthographic and perspective settings. For Applications
    controlled by application detection, changing this is inadvisable.
    Applications that use context creation attributes to enable 3D
    Vision may have to set the method explicitly.

    3. Should descriptions of how these settings are used by 3D Vision
    be included here?

    RESOLVED: No. This functionality is extensively documented in
    existing 3D Vision documents such as "NVIDIA 3D Vision Automatic Best
    Practices". Merging these explanations into the OpenGL document and
    maintaining them won't add any additional clarity to the functionality
    and may limit the underlying implementation.


    4. What is the behavior of GL calls that may use a 3D Vision surface
    such as ReadPixels, CopyTex[Sub]Image, and BlitFramebuffer?

    RESOLVED: The current implementation makes calls that expect a
    single surface by definition such as ReadPixels read from the left
    buffer only. Calls that permit the number of surfaces to be
    abstracted behind objects that have no direct interaction with the
    user application such as CopyTexImage and BlitFramebuffer sample
    from both surfaces of the source when available to copy to the
    corresponding destination surfaces.

    The intent of this extension is to provide a 3D Vision support
    that is as transparent to the developer as possible. Introducing the
    ability to select which buffer is read from breaks the notion of the
    application that what was developed as a single buffer is
    represented as two surfaces when 3D Vision is enabled. Furthermore,
    this matches behavior of 3D Vision enabled applications that do not
    make use of this extension.

    It's possible that this approach may cause problems with particular
    applications that make meaningful use of ReadPixels. Such
    incidents are not expected to be common. Where they arise, the 3D
    Vision implementation may change to accomodate them. It is in
    keeping with one purpose of this extension to test new development
    with 3D Vision to identify interaction problems before it becomes
    publicly available. By keeping with the current implementation,
    these conflicts can be identified and resolved beforehand. This is
    one such conflict that may be identified and addressed either by the
    application developer or through communication with the extension
    implementor.

    Because the implementation may change according to future
    requirements, this extension makes no changes to the corresponding
    documentation in the specification to enable future implementation
    changes that may do something different. Conceptually, the behavior
    of this functionality does not change any more than that of the draw
    calls.

Revision History

    Rev.    Date        Author      Changes
    ----  ------------- ---------   ----------------------------------------
      5   05 Oct 2012   groth       Add issue addressing ReadPixels.
      4   15 Dec 2011   groth       Refine default values. Add invalid value errors
      3   02 Dec 2011   groth       Restore method settings in limited form
      2   11 Nov 2011   groth       Remove method query and settings
      1   08 Nov 2011   groth       Initial version

