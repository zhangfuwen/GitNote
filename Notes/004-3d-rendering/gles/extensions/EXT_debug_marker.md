# EXT_debug_marker

Name

    EXT_debug_marker

Name Strings

    GL_EXT_debug_marker

Contributors

    Seth Sowerby
    Benj Lipchak

Contact

    Benj Lipchak, Apple (lipchak 'at' apple.com)

Status
    
    Complete

Version

    Date: October 7, 2013
    Revision: 3

Number

    OpenGL Extension #440
    OpenGL ES Extension #99

Dependencies
    
    Requires OpenGL ES 1.1.

    Written based on the wording of the OpenGL ES 2.0.25 Full Specification
    (November 2, 2010).

Overview

    This extension defines a mechanism for OpenGL and OpenGL ES applications to
    annotate their command stream with markers for discrete events and groups 
    of commands using descriptive text markers. 
    
    When profiling or debugging such an application within a debugger or 
    profiler it is difficult to relate the commands within the command stream 
    to the elements of the scene or parts of the program code to which they 
    correspond. Markers help obviate this by allowing applications to specify 
    this link.
    
    The intended purpose of this is purely to improve the user experience 
    within OpenGL and OpenGL ES development tools.
    
New Procedures and Functions

    void InsertEventMarkerEXT(sizei length, const char *marker);
    void PushGroupMarkerEXT(sizei length, const char *marker);
    void PopGroupMarkerEXT();

New Tokens

    None

Additions to Chapter 2 of the OpenGL ES 2.0 Specification (OpenGL Operation)

    None

Additions to Chapter 3 of the OpenGL ES 2.0 Specification (Rasterization)

    None

Additions to Chapter 4 of the OpenGL ES 2.0 Specification (Per-Fragment
Operations and the Framebuffer)

    None

Additions to Chapter 5 of the OpenGL ES 2.0 Specification (Special Functions)

    Add a new section titled Debug Makers

    Debug markers provide a method for annotating a command stream with markers 
    for discrete events and groups of commands using a descriptive text marker.
    These names may then be used by a tool such as a debugger or profiler to 
    label the command stream.
    
    The command
    
        void InsertEventMarkerEXT(sizei length, const char *marker);
        
    inserts an event marker string <marker> into the command stream. <length> 
    specifies the length of the string passed in <marker>. If <marker> is a 
    null-terminated string then <length> should not include the terminator. 
    If <length> is 0 then <marker> is assumed to be null-terminated.

    The command
    
        void PushGroupMarkerEXT(sizei length, const char *marker);
        
    pushes a group marker string <marker> into the command stream. <length> 
    specifies the length of the string passed in <marker>. If <marker> is a 
    null-terminated string then <length> should not include the terminator. If 
    <length> is 0 then <marker> is assumed to be null-terminated. If <marker> 
    is null then an empty string is pushed on the stack.
        
    The command
    
        void PopGroupMarkerEXT();
        
    pops the most recent group marker. If there is no group marker to pop then 
    the PopGroupMarkerEXT command is ignored.
    
    Group markers are strictly hierarchical. Group marker sequences may be 
    nested within other group markers but can not overlap.

Additions to Chapter 6 of the OpenGL ES 2.0 Specification (State and State
Requests)

    None

Errors

    None

New State

    None

New Implementation Dependent State

    None

Issues

    (1) Should the extension provide a method for querying markers?
    
    No.
    
    A great deal of the value of debug markers is the 'when' as well as the 
    'what' - seeing the debug markers within the stream of all OpenGL ES 
    commands.  This value is only available to a tool intercepting the command 
    stream - it is not available from querying the markers.  However, the 
    ability to query markers would make them available to developer tools 
    attaching to an already running application.
    
    Querying the markers could be useful for applications to be able to dump 
    their marker stack to their own logs.  However, this functionality does not 
    require an extension as applications can implement their own marker stacks 
    within their code independent of OpenGL ES.
    
    (2) Should a query exist for the current marker stack depth?
    
    No.
    
    This would be useful if markers are queryable but not otherwise.
        
    (3) Should PushGroupMarkerEXT & PopGroupMarkerEXT return the marker 
        stack depth?
    
    No.
     
    This would be useful if markers are queryable but not otherwise.
        
    (4) How should a null-string passed to PushGroupMarkerEXT be treated?

    Resolved: Push an empty string.
    
    The two possibilities are to push an empty string onto the marker stack or 
    to ignore the call to PushGroupMarkerEXT.  Pushing an empty string 
    maintains the marker stack depth expected by the calling application.
    
    (5) Should the extension support printf-style formatting?

    Resolved: No.

    Providing printf-style formatting would impose a much greater burden on the 
    extension in terms of error checking the format string and arguments.  
    Likely all languages capable of calling OpenGL ES have convenient 
    capabilities for formatting strings so it is acceptable to rely on those.

Revision History

    Date 01/17/2011
    Revision: 1
       - draft proposal

    Date 07/22/2011
    Revision: 2
       - rename from APPLE to EXT

    Date 10/07/2013
    Revision: 3
       - Add support for desktop OpenGL
