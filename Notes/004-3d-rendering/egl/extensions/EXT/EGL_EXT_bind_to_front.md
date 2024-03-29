# EXT_bind_to_front

Name

  EXT_bind_to_front

Name Strings

  EGL_EXT_bind_to_front


Contributors

  Daniel Herring

Contacts
  
  Daniel Herring, Core Avionics & Industrial Inc., daniel dot herring at ch1group dot com

Status

  Complete

Version

  Version 1.0, February 21, 2017

Number

  EGL Extension #121

Dependencies

  Requires EGL 1.2.

Overview

  This extension allows for using double buffered Pbuffers for rendering to textures, by 
  allowing a new enumeration to be used in eglBindTexImage. EGL_FRONT_BUFFER_EXT is used
  to denote reading the textures data from the front buffer of a double buffered Pbuffer.

New Types

  None

New Procedures and Functions

  None

New Tokens
  
  EGL_FRONT_BUFFER_EXT          0x3464     

Modify Section 3.6.1 of the EGL 1.2 Specification, paragraph 1
  Add EGL_FRONT_BUFFER_EXT to list of acceptable values for buffer attribute.

Modify Section 3.6.2 of the EGL 1.2 Specification, paragraph 5
  Add EGL_FRONT_BUFFER_EXT to list of acceptable values for buffer attribute.

Modify Section 3.5.2 of the EGL 1.2 Specification, paragraph 4 
  Add EGL_RENDER_BUFFER to list of attributes which can be accepted in attrib_list.
  
Add to Section 3.5.2 of the EGL 1.2 Specification
  EGL_RENDER_BUFFER specifies the number of color buffers which should be useable by the
  client API rendering to the Pbuffer. If its value is EGL_SINGLE_BUFFER, then there is 
  one color buffer the client APIs will render to directly. If its value is EGL_BACK_BUFFER,
  then there are at least two color buffers the client API can render to. eglSwapBuffers 
  is used to switch which color buffer is currently being rendered to. By default, all 
  client APIs should render into the back buffer. The default value of EGL_RENDER_BUFFER
  is EGL_SINGLE_BUFFER.

Modify Section 2.2.2 of the EGL 1.2 Specification, paragraph 2
    Pbuffer surfaces have a back buffer but no associated window, so the back buffer 
    need not be copied.
  Change to:
    Pbuffer surfaces have no associated window, and include a back buffer, used by 
    default, for rendering to by the client API. Pbuffers may have a front buffer 
    used during render to texture operations to provide a read only texture which may
    be used while the back buffer is being rendered to.
 
Modify Section 3.5.6 
    Querying EGL RENDER BUFFER returns the buffer which client API rendering
    is requested to use. For a window surface, this is the same attribute value
    specified when the surface was created. For a Pbuffer surface, it is always
    EGL BACK BUFFER. For a pixmap surface, it is always EGL SINGLE BUFFER. To
    determine the actual buffer being rendered to by a context, call eglQueryContext
    (see section 3.7.4).

  Change to:
    Querying EGL RENDER BUFFER returns the buffer which client API rendering
    is requested to use. For a window surface or Pbuffer surface, this is the 
    same attribute value specified when the surface was created. For a pixmap 
    surface, it is always EGL SINGLE BUFFER. To determine the actual buffer being
    rendered to by a context, call eglQueryContext (see section 3.7.4).

Revision History
  Version 1.0, 21/02/2017 - Initial Version
  
Question:
  What if the implementation  already uses double buffering for single buffered PBuffers? 
  Such as when an implementation must insert a resolve to a texture instead of being 
  able to use the same memory.
Answer:
  EGL_BACK_BUFFER would still refer to the resolved buffer. But in the case of the a 
  double buffer EGL_FRONT_BUFFER should be used to reference the resolved buffer instead
  of EGL_BACK_BUFFER as in this case the user has specified 2 buffers and knows they want 
  the resolved buffer.
  In the double buffer case where only a back draw and a front resolved buffer is used 
  the implementation may not support EGL_BACK_BUFFER and should generate an EGL_BAD_
  PARAMETER error when not supporting EGL_BACK_BUFFER texture binding.
