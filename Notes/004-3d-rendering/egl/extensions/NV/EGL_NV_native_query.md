# NV_native_query

Name

    NV_native_query

Name Strings

    EGL_NV_native_query

Contributors

    Mathias Heyer, NVIDIA
    Daniel Kartch, NVIDIA
    Peter Pipkorn, NVIDIA
    Acorn Pooley, NVIDIA
    Greg Roth, NVIDIA
    
Contacts

    Peter Pipkorn, NVIDIA Corporation (ppipkorn 'at' nvidia.com)

Status

    Complete

Version

    Version 0.4, 25 Sept, 2012

Number

    EGL Extension #45

Dependencies

    Requires EGL 1.0

    This extension is written against the wording of the EGL 1.4
    Specification.

Overview

    This extension allows an application to query which native display,
    pixmap and surface corresponds to a EGL object. 

New Procedures and Functions

    EGLBoolean eglQueryNativeDisplayNV(
                            EGLDisplay dpy,
                            EGLNativeDisplayType* display_id);

    EGLBoolean eglQueryNativeWindowNV(
                            EGLDisplay dpy,
                            EGLSurface surf,
                            EGLNativeWindowType* window);

    EGLBoolean eglQueryNativePixmapNV(
                            EGLDisplay dpy,
                            EGLSurface surf,
                            EGLNativePixmapType* pixmap);

Changes to Chapter 3 of the EGL 1.4 Specification (EGL Functions and Errors)
    
    In Chapter 3.2, after the description of eglInitialize and before the 
    description of eglTerminate insert

        "While initialized, the native display that corresponds to an
         EGLDisplay can retrieved by
        
            EGLBoolean eglQueryNativeDisplayNV(
                            EGLDisplay dpy,
                            EGLNativeDisplayType* display_id);
         
         If the <dpy> is a valid and initialized EGLDisplay, EGL_TRUE
         will be returned and the native display handle will be written
         to <display_id>. Otherwise EGL_FALSE will be returned and the
         contents of <display_id> are left untouched. If the <dpy> is
         not valid, an EGL_BAD_DISPLAY error will be generated. If <dpy>
         is not initialized, an EGL_NOT_INITIALIZED error will be
         generated. If <display_id> is NULL, an EGL_BAD_PARAMETER error
         will be generated.

    In Chapter 3.5 Rendering Surfaces, after section 3.5.1 insert
      
         "The native window that corresponds to an EGLSurface can be
          retrieved by

            EGLBoolean eglQueryNativeWindowNV(
                            EGLDisplay dpy,
                            EGLSurface surf,
                            EGLNativeWindowType* win);

          The corresponding native window will be written to <win>, 
          and EGL_TRUE will be returned. If the call fails, EGL_FALSE
          will be returned, and content of <win> will not be modified.
          If <dpy> is not a valid EGLDisplay, an EGL_BAD_DISPLAY error
          will be generated. If <dpy> is not initialized, an EGL_NOT_-
          INITIALIZED error will be generated. If <surf> is not a valid
          EGLSurface, or <surf> does not have a corresponding native
          window, an EGL_BAD_SURFACE error will be generated." If <win>
          is NULL, an EGL_BAD_PARAMETER error will be generated.

     After section 3.5.4 Creating Native Pixmap Rendering Surfaces insert
      
         "The native pixmap that corresponds to an EGLSurface can be
          retrieved by

            EGLBoolean eglQueryNativePixmapNV(
                            EGLDisplay dpy,
                            EGLSurface surf,
                            EGLNativePixmapType* pixmap);

          The corresponding native pixmap will be written to <pixmap>, 
          and EGL_TRUE will be returned. If the call fails, EGL_FALSE
          will be returned, and the content of <pixmap> will not be
          modified. If <dpy> is not a valid EGLDisplay, an EGL_BAD_-
          DISPLAY error will be generated. If <dpy> is not initialized,
          an EGL_NOT_INITIALIZED error will be generated. If <surf> is
          not a valid EGLSurface, or <surf> does not have a corresponding
          native pixmap, an EGL_BAD_SURFACE error will be generated." If
          <pixmap> is NULL, an EGL_BAD_PARAMETER error will be
          generated.

Issues
    
Revision History
#4 (Greg Roth, Sept 25, 2012)
   - Further document all potential errors for all functions

#3 (Daniel Kartch, August 30, 2011)
   - Add restriction that EGLDisplay be initialized

#2 (Peter Pipkorn, December 16, 2009)
   - Minor cleanup

#1 (Peter Pipkorn, December 15, 2009)
   - First Draft

