# OVR_multiview2

Name

    OVR_multiview2

Name Strings

    GL_OVR_multiview2

Contact

    Cass Everitt, Oculus (cass.everitt 'at' oculus.com)

Contributors

    John Carmack, Oculus
    Jan Paul van Waveren, Oculus
    Jan-Harald Fredriksen, ARM
    Sam Holmes, Qualcomm
    Daniel Koch, NVIDIA

Status

    Incomplete.

Version

    Last Modified Date:  October 19, 2018
    Author Revision: 0.5

Number

    OpenGL Extension #479
    OpenGL ES Extension #242

Dependencies

    OpenGL 3.0 or OpenGL ES 3.0 is required.

    This extension is written against the OpenGL ES Shading Language 3.00.04
    specification.

    This extension requires OVR_multiview.

Overview

 
	This extension relaxes the restriction in OVR_multiview that only gl_Position
	can depend on ViewID in the vertex shader.  With this change, view-dependent
	outputs like reflection vectors and similar are allowed.

New Tokens

    None.

New Procedures and Functions

    None.

Modifications to The OpenGL ES Shading Language Specification, Version 3.00.04

	Including the following line in a shader can be used to control the language
	features described in this extension:

      #extension GL_OVR_multiview2 : <behavior>

    where <behavior> is as specified in section 3.5.

    A new preprocessor #define is added to the OpenGL ES Shading Language:

      #define GL_OVR_multiview2 1

    If the OVR_multiview2 extension is enabled, the OVR_multiview extension
    is also implicitly enabled.

    Modifications to Section 7.1 "Built-in Language Variables" (as modified by
    OVR_multiview):

    Delete the paragraph which states:

    It is a compile- or link-time error if any output variable other
    than gl_Position is statically dependent on gl_ViewID_OVR. If an
    output variable other than gl_Position is dynamically dependent on
    gl_ViewID_OVR, the values are undefined.

Errors

    None.

Issues

    (1) Does enabling GL_OVR_multiview2 automatically enable GL_OVR_multiview?

    RESOLVED: Yes. This was not specified in early drafts of the extension,
    but this is the behavior of multiple implementations.

Revision History

      Rev.    Date    Author    Changes
      ----  --------  --------  -----------------------------------------
      0.1   03/18/15  cass      Initial draft
      0.5   10/19/18  dgkoch    Add standard boiler plate shader extension language.
                                Add issue 1 and clarify that this extension
                                implicitly enables OVR_multiview.
