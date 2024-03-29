# HI_colorformats

Name

    HI_colorformats

Name Strings

    EGL_HI_colorformats

Contributors

    Guillaume Portier

Contacts

    HI support. (support_renderion 'at' hicorp.co.jp)

Status

    Shipping (Revision 2)

Version

    Last Modified Date: June 7, 2010
    Revision 2.1

Number

    EGL Extension #25

Dependencies

    These extensions are written against the wording of the EGL 1.4
    Specification.

Overview

    The extensions specified in this document provide a mechanism for
    creating ARGB color-buffers, as opposed to the default RGBA
    format used by other EGL configurations.

New Types
   
    None.

New Procedures and Functions

    None.

New Tokens

      Accepted in the <attrib_list> parameter of eglChooseConfig.

          EGL_COLOR_FORMAT_HI				0x8F70

      Accepted as a value for the EGL_COLOR_FORMAT_HI token:

          EGL_COLOR_RGB_HI				0x8F71
          EGL_COLOR_RGBA_HI				0x8F72
          EGL_COLOR_ARGB_HI				0x8F73


      The default value for EGL_COLOR_FORMAT_HI is EGL_DONT_CARE.
      If EGL_COLOR_FORMAT_HI is used with a value other than
      EGL_DONT_CARE, EGL_COLOR_RGB_HI, EGL_COLOR_RGBA_HI or
      EGL_COLOR_ARGB_HI then an EGL_BAD_ATTRIBUTE is generated.

      EGL_COLOR_RGB_HI, EGL_COLOR_RGBA_HI and EGL_COLOR_ARGB_HI 
      specify the order of the color components in the color-buffer.
      EGL_COLOR_RGB_HI must be used only with configurations having no
      alpha component, currently only 565.

      EGL_COLOR_RGBA_HI and EGL_COLOR_ARGB_HI must be used with
      configurations having an alpha component. Currently available
      configurations are:
          - 4444
          - 5551
          - 8888
      Currently EGL_COLOR_ARGB_HI can be used only with the
      8888 configuration.

      If the value used for EGL_COLOR_FORMAT_HI does not match
      the other specified attributes of the EGL config then an
      EGL_BAD_MATCH is generated.

      When EGL_COLOR_FORMAT_HI is unspecified or equals EGL_DONT_CARE,
      eglChooseConfig returns configurations having all kinds of color
      component ordering.

      If eglCreatePixmapSurface is called with a configuration that
      does not match the pixmap's native ordering then an EGL_BAD_MATCH
      is generated. To be sure to call eglCreatePixmapSurface with a
      compatible configuration, the application should either parse
      the <configs> list returned by eglChooseConfig or explicitly specify
      EGL_COLOR_FORMAT_HI to match the pixmap native format.

Example

    EGLint attrib_list[] = {
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_SURFACE_TYPE, EGL_PIXMAP_BIT,

        #ifdef USE_ARGB           // Specifying ARGB as a color format
        EGL_COLOR_FORMAT_HI, EGL_COLOR_ARGB_HI,
        #else                     // Specifying RGBA as a color format
        EGL_COLOR_FORMAT_HI, EGL_COLOR_RGBA_HI,
        #endif

        EGL_NONE
    };

    // Get one of the configuration matching the config_list
    eglChooseConfig(dpy, attrib_list, &config, 1, &num_config);

    // Create the pixmap
    eglCreatePixmapSurface(dpy, config[0], pixmap, NULL);


Issues

    None.


Revision History


    June 7, 2010 (r2.1)
        - Corrected mistaken description of EGL_COLOR_FORMAT_HI as
          attribute for eglCreatePixmapSurface. Clarified other text.

    June 16, 2009 (r2)
	- Split HI_clientpixmap into two different extensions:
          -  HI_colorformats
          -  HI_clientpixmap

    March 3, 2009 (r1)
