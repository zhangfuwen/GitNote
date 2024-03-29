# EXT_multiview_timer_query

Name

    EXT_multiview_timer_query

Name Strings

    GL_EXT_multiview_timer_query

Contact

    Robert Menzel, NVIDIA Corporation (rmenzel 'at' nvidia.com)

Contributors

    Pat Brown, NVIDIA Corporation
    James Helferty, NVIDIA Corporation
    Kedarnath Thangudu, NVIDIA Corporation

Status

    Complete.

Version

    Last Modified Date:  May 13, 2019
    Author Revision: 1

Number

    OpenGL Extension #536
    OpenGL ES Extension #317

Dependencies

    OpenGL 4.0 or OpenGL ES 3.2 are required.

    This extension is written against the OpenGL 4.6 specification
    (Core Profile) (February 2, 2019) and OpenGL ES 3.2 specification
    (February 2, 2019).

    OVR_multiview is required.

Overview

    OVR_multiview introduced multiview rendering to OpenGL and OpenGL ES.
    This extension removes one of the limitations of the OVR_multiview 
    extension by allowing the use of timer queries during multiview rendering. 
    OVR_multiview does not specify defined behavior for such usage
    (if implemented in OpenGL or if EXT_disjoint_timer_query is present).
    
    
New Tokens

    None.

    
New Procedures and Functions

    None.
    
    
Modifications to Chapter 4 of the OpenGL 4.6 Specification (Event Model)

    Remove the following error from section 4.3 (Time Queries) which has
    been added by OVR_multiview:
    
    "Queries where BeginQuery or EndQuery is called with a target of
    TIME_ELAPSED, or a if QueryCounter is called with a target of TIMESTAMP
    return undefined values if the draw framebuffer is multiview at any
    point during their execution."
    
    
Modifications to Chapter 9 of the OpenGL ES 3.2 Specification as well as
Chapter 9 of the OpenGL 4.6 Specification (Framebuffers and Framebuffer 
Objects) 
    
    Modify the subsection 9.2.2.2 (Multiview Images) which has been added by 
    OVR_multiview:
    
    Remove from the list of restrictions ("In this mode there are several 
    restrictions:") the following item:

    "- no timer queries (section 4.3)".
      
    
Issues

    (1) This extension is based on an OVR extension, why call it EXT?
    
    While started as a single vendor extension, OVR_multiview and OVR_multiview2
    are already supported by multiple vendors. This new extension also has the 
    support from multiple vendors to be specified as EXT.
    
    
Revision History

      Rev.    Date    Author    Changes
      ----  --------  --------  -----------------------------------------
        1   05/13/19  rmenzel   Initial version.
