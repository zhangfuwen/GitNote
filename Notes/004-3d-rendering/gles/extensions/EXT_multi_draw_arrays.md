# 


Name

    EXT_multi_draw_arrays
    SUN_multi_draw_arrays

Name Strings

    GL_EXT_multi_draw_arrays
    GL_SUN_multi_draw_arrays

    (Note: this extension has been promoted from SUN to EXT status.
    Implementations should advertise both name strings, and both EXT
    and SUN versions of the new GL functions should be provided).

Contact

    Ron Bielaski, Sun (Ron.Bielaski 'at' eng.sun.com)
    Jack Middleton, Sun (Jack.Middleton 'at' eng.sun.com)

Status

    Shipping

Version

    Version 4, June 18, 2013

Number

    148
    OpenGL ES Extension #69

Dependencies

    OpenGL 1.1 is required. The language is written against the OpenGL 1.2
    specification.

Overview

    These functions behave identically to the standard OpenGL 1.1 functions
    glDrawArrays() and glDrawElements() except they handle multiple lists of
    vertices in one call. Their main purpose is to allow one function call
    to render more than one primitive such as triangle strip, triangle fan,
    etc.

New Procedures and Functions

    void glMultiDrawArraysEXT( GLenum mode,
			       const GLint *first,
			       const GLsizei *count,
			       GLsizei primcount)
    Parameters
    ----------
	mode		Specifies what kind of primitives to
			render. Symbolic constants GL_POINTS,
			GL_LINE_STRIP, GL_LINE_LOOP, GL_LINES,
			GL_TRIANGLE_STRIP, GL_TRIANGLE_FAN,
			GL_TRIANGLES, GL_QUAD_STRIP, GL_QUADS,
			and GL_POLYGON are accepted.

	first		Points to an array of starting indices in
			the enabled arrays.

	count		Points to an array of the number of indices
			to be rendered.

	primcount	Specifies the size of first and count


    void glMultiDrawElementsEXT( GLenum mode,
				 GLsizei *count,
				 GLenum type,
				 const void * const *indices,
				 GLsizei primcount)

    Parameters
    ----------
	mode		Specifies what kind of primitives to render.
			Symbolic constants GL_POINTS, GL_LINE_STRIP,
			GL_LINE_LOOP, GL_LINES, GL_TRIANGLE_STRIP,
			GL_TRIANGLE_FAN, GL_TRIANGLES, GL_QUAD_STRIP,
			GL_QUADS, and GL_POLYGON are accepted.

	count		Points to and array of the element counts

	type		Specifies the type of the values in indices.
			Must be  one  of GL_UNSIGNED_BYTE,
			GL_UNSIGNED_SHORT, or GL_UNSIGNED_INT.

	indices		Specifies a  pointer to the location where
			the indices are stored.

	primcount	Specifies the size of the count array

New Tokens

    None

Additions to Chapter 2 of the 1.2 Specification (OpenGL Operation)

    Section 2.8 Vertex Arrays:

    The command

	void glMultiDrawArraysEXT( GLenum mode,
				   const GLint* first,
				   const GLsizei *count,
				   GLsizei primcount)

    Behaves identically to DrawArrays except that a list of arrays is
    specified instead. The number of lists is specified in the primcount
    parameter. It has the same effect as:

	for(i=0; i<primcount; i++) {
	   if (*(count+i)>0) DrawArrays(mode, *(first+i), *(count+i));
	}

    The command

	void glMultiDrawElementsEXT( GLenum mode,
				     GLsizei *count,
				     GLenum type,
				     const void **indices,
				     GLsizei primcount)

    Behaves identically to DrawElements except that a list of arrays is
    specified instead. The number of lists is specified in the primcount
    parameter. It has the same effect as:

	for(i=0; i<primcount; i++) {
	    if (*(count+i)>0) DrawElements(mode, *(count+i), type,
					   *(indices+i));
	}

Additions to Chapter 3 of the 1.2 Specification (Rasterization)

    None

Additions to Chapter 4 of the 1.2 Specification (Per-Fragment Operations and

    None

Additions to Chapter 5 of the 1.2 Specification (Special Functions)

    None

Additions to Chapter 6 of the 1.2 Specification (State and State Requests)

    None

Additions to the GLX / WGL / AGL Specifications

    None

GLX Protocol

    None

Errors

    GL_INVALID_ENUM is generated if <mode> is not an accepted value.

    GL_VALUE is generated if <primcount> is negative.

    GL_INVALID_OPERATION is generated if glMultiDrawArraysEXT or
    glMultiDrawElementsEXT is executed between the execution of glBegin
    and the corresponding glEnd.

New State

    None

OpenGL ES interactions
    This extension can also be implemented against OpenGL ES 1.x or
	OpenGL ES 2.0. In those cases, remove references to glBegin, glEnd,
	and to GL_QUAD_STRIP, GL_QUADS, and GL_POLYGON.


Revision History

    Version 5, 2013/09/08 (Jon Leech) - Changed GLvoid -> void
        (Bug 10412).
    Version 4, 2013/06/18 (Jon Leech) - Added 'const' to
	MultiDrawElementsEXT, too, based on feedback from Mesa.
    Version 3, 2010/08/06 - Added 'const' to MultiDrawArraysEXT pointer
	arguments to match the core GL entry point.
    30/09/09 - Added fields from the new extension template and
	interactions with OpenGL ES.
    6/24/99 - Added fields from the new extension template.
