# Android Kotlin开发

## Logcat

 adb logcat -v color,threadtime -s MyTAG:D
 `adb logcat -s`等价于`adb logcat *:S`，即所有tag为Silent。


## am start




# Gradle

gradle版本8.2
kotlin version:

```kotlin
id("org.jetbrains.kotlin.android") version "2.1.20" apply false
```

# kotlin

# kotlin show preview 

**build.gradle**

```Kotlin
buildFeatures {
compose true
}
```

# how to use a Compose View and pass it as an argument of android's traditional addView

If you want to use a Compose View and pass it as an argument to the traditional `addView` method in Android, you can follow these steps:

1. Create a Compose View using the `AndroidView` composable function. This function allows you to embed an Android View inside a Compose UI hierarchy.

```kotlin
@Composable
fun MyComposeView() {
    AndroidView(factory = { context ->
        // Create and return your custom Android View
        MyCustomView(context).apply {
            // Set any necessary properties or listeners
        }
    })
}
```

In the above example, `MyCustomView` is an Android View that you want to embed in a Compose UI.

2. In your traditional Android code, locate the parent view where you want to add the Compose View.

```kotlin
val parentView = findViewById<ViewGroup>(R.id.parentView)
```

3. Create an instance of the `composable` function by wrapping your Compose View.

```kotlin
val myComposeView = @Composable { MyComposeView() }
```

4. Inflate the Compose View using the `setContent` function and add it to the parent view using the `addView` method.

```kotlin
ViewCompositionStrategy.DisposeOnViewTreeLifecycleDestroyed.on(parentView).apply {
    setContent {
        myComposeView()
    }
}
```

In the above code, `ViewCompositionStrategy.DisposeOnViewTreeLifecycleDestroyed` makes sure that the Compose View is properly disposed of when the view tree is destroyed.

5. If needed, you can update the Compose View by recomposing it with new data.

```kotlin
myComposeView.invalidate()
```

This step is optional and should be used if you want to update the Compose View dynamically.

By following these steps, you can create a Compose View and pass it as an argument to the `addView` method in traditional Android development.

# how to call android api to recogize speech?

To call the Android Speech Recognition API in Kotlin, you can use the `RecognizerIntent` class along with an `ActivityResultLauncher` to handle the result. Here's an example code snippet that demonstrates how to implement speech recognition:

```kotlin
import android.app.Activity
import android.content.Intent
import android.speech.RecognizerIntent
import android.speech.RecognizerIntent.EXTRA_LANGUAGE_MODEL
import android.speech.RecognizerIntent.EXTRA_PROMPT
import android.speech.RecognizerIntent.RESULTS_RECOGNITION
import android.speech.RecognizerIntent.ACTION_RECOGNIZE_SPEECH
import android.os.Bundle
import android.speech.SpeechRecognizer
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity
import androidx.activity.result.contract.ActivityResultContracts

class MainActivity : AppCompatActivity() {

    private lateinit var speechRecognizer: SpeechRecognizer
    private lateinit var startSpeechRecognition: ActivityResultLauncher<Intent>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize the SpeechRecognizer
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)

        // Set up the ActivityResultLauncher
        startSpeechRecognition = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                val data: Intent? = result.data
                val matches: ArrayList<String>? = data?.getStringArrayListExtra(EXTRA_RESULTS)
                if (!matches.isNullOrEmpty()) {
                    val recognizedText = matches[0]
                    // Handle the recognized text here
                }
            }
        }

        // Start speech recognition when the button is clicked
        val button: Button = findViewById(R.id.button)
        button.setOnClickListener {
            startSpeechRecognition()
        }
    }

    private fun startSpeechRecognition() {
        val intent = Intent(ACTION_RECOGNIZE_SPEECH)
        intent.putExtra(EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
        intent.putExtra(EXTRA_PROMPT, "Speak now...")

        startSpeechRecognition.launch(intent)
    }

    override fun onDestroy() {
        super.onDestroy()
        speechRecognizer.destroy()
    }
}
```

In this example, we create a `SpeechRecognizer` instance in the `onCreate` method and initialize it using the `SpeechRecognizer.createSpeechRecognizer` method.

We set up an `ActivityResultLauncher` called `startSpeechRecognition` using the `registerForActivityResult` method. The launcher listens for the result of the speech recognition activity.

When the button is clicked, the `startSpeechRecognition` function is called. It creates an `Intent` with the action `ACTION_RECOGNIZE_SPEECH` and sets extras such as `EXTRA_LANGUAGE_MODEL` and `EXTRA_PROMPT`. The `startSpeechRecognition` launcher is then invoked with the intent using `launch(intent)`.

Once the speech recognition activity completes, the result is received in the `ActivityResultCallback` of the `startSpeechRecognition` launcher. The recognized speech is extracted from the result and can be handled as needed.

Remember to include the necessary permissions in your app's manifest file:

```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
```

Also, note that the speech recognition functionality requires the Google app or Google Play Services to be installed on the device.

# example code to call android tts api in kotlin

Certainly! Here's an example code snippet to call the Android Text-to-Speech (TTS) API in Kotlin:

```kotlin
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeech.OnInitListener
import android.speech.tts.TextToSpeech.OnUtteranceCompletedListener
import android.speech.tts.UtteranceProgressListener
import android.content.Context
import android.os.Bundle
import android.speech.tts.TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID
import java.util.*

class MainActivity : AppCompatActivity(), OnInitListener {

    private lateinit var tts: TextToSpeech

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tts = TextToSpeech(this, this)
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = tts.setLanguage(Locale.US)
            
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                // Language data is missing or the language is not supported.
                // Handle the error case here.
            } else {
                // TTS engine initialized successfully.
                // Call the function to speak the desired text.
                speakText("Hello, World!")
            }
        } else {
            // TTS initialization failed. Handle the error case here.
        }
    }

    private fun speakText(text: String) {
        val utteranceId = UUID.randomUUID().toString()
        
        tts.setOnUtteranceCompletedListener(object : UtteranceProgressListener() {
            override fun onDone(utteranceId: String) {
                // Speech completed.
            }

            override fun onError(utteranceId: String) {
                // Speech error occurred.
            }

            override fun onStart(utteranceId: String) {
                // Speech started.
            }
        })
        
        val params = HashMap<String, String>()
        params[KEY_PARAM_UTTERANCE_ID] = utteranceId

        tts.speak(text, TextToSpeech.QUEUE_FLUSH, params)
    }

    override fun onDestroy() {
        super.onDestroy()
        tts.stop()
        tts.shutdown()
    }
}
```

In this example, we create an instance of the `TextToSpeech` class in the `onCreate` method. We implement the `OnInitListener` interface to handle the initialization process. Once the TTS engine is successfully initialized, we set the language to US English and call the `speakText` function to speak the desired text.

The `speakText` function sets an `OnUtteranceCompletedListener` to handle the completion of speech. It generates a unique utterance ID for each speech request and passes it as a parameter using a `HashMap`. Finally, we call the `speak` method of the `TextToSpeech` instance to initiate the speech synthesis.

Remember to include the necessary permissions and dependencies in your app's manifest file.

# android add menu button to nav

To add a menu button to the navigation bar in an Android application, you can follow these steps:

1. Create a menu resource file: Create an XML file in the `res/menu` directory of your Android project. This file will define the items for your menu. For example, let's name the file `menu_main.xml` and define a single item:

   ```xml
   <!-- menu_main.xml -->
   <menu xmlns:android="http://schemas.android.com/apk/res/android">
       <item
           android:id="@+id/menu_item"
           android:title="Menu Item" />
   </menu>
   ```

2. Inflate the menu in your activity or fragment: In your activity or fragment code, override the `onCreateOptionsMenu` method and inflate the menu resource:

   ```kotlin
   override fun onCreateOptionsMenu(menu: Menu): Boolean {
       menuInflater.inflate(R.menu.menu_main, menu)
       return true
   }
   ```

3. Handle menu item selection: Override the `onOptionsItemSelected` method to handle the selection of the menu item:

   ```kotlin
   override fun onOptionsItemSelected(item: MenuItem): Boolean {
       when (item.itemId) {
           R.id.menu_item -> {
               // Handle menu item click
               return true
           }
           else -> return super.onOptionsItemSelected(item)
       }
   }
   ```

4. Display the menu button: To display the menu button on the navigation bar, you need to enable the option on the device. Depending on the device and Android version, the process may vary. Commonly, you can either have a hardware menu button on older devices or a software-based button on the navigation bar for newer devices.

   For devices with a dedicated hardware menu button, the menu will automatically appear when the button is pressed.

   For devices without a hardware menu button, you can use the following code snippet to display the software-based menu button on the navigation bar:

   ```kotlin
   override fun onCreate(savedInstanceState: Bundle?) {
       super.onCreate(savedInstanceState)
       setContentView(R.layout.activity_main)

       // Show the software-based menu button on the navigation bar
       val flags = View.SYSTEM_UI_FLAG_LAYOUT_STABLE or
               View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN or
               View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
       window.decorView.systemUiVisibility = flags
   }
   ```

   This code sets the appropriate system UI flags to display the menu button on the navigation bar.

With these steps, you can add a menu button to the navigation bar in your Android application. When the menu button is clicked, the `onOptionsItemSelected` method will be called, allowing you to handle the menu item selection as desired.

# how to make textview scrollable

To make a `TextView` scrollable in Android, you can use a `ScrollView` container. Here's an example of how to achieve that:

```xml
<ScrollView
    android:layout_width="match_parent"
    android:layout_height="match_parent">

    <TextView
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="Your long text here"
        android:maxLines="1"
        android:scrollbars="vertical"
        android:scrollbarStyle="outsideOverlay"
        android:fadeScrollbars="false"
        android:id="@+id/scrollable_text_view" />
        
</ScrollView>
```

In this example, the `ScrollView` wraps around the `TextView`, allowing the text to be scrollable if it exceeds the available space.

The `android:scrollbars` attribute is set to `"vertical"` to enable vertical scrolling. You can also use `"horizontal"` for horizontal scrolling.

To customize the appearance of the scrollbars, you can use the `android:scrollbarStyle` attribute. In this example, `"outsideOverlay"` is used, which displays the scrollbars outside the content area. Other options include `"insideOverlay"` and `"insideInset"`. You can experiment with these styles to achieve the desired visual effect.

The `android:fadeScrollbars` attribute is set to `"false"` to ensure that the scrollbars are always visible. If you want the scrollbars to fade out when not in use, you can set this attribute to `"true"`.

Note that the `TextView` should have sufficient content to exceed the available space for scrolling to be enabled. If the text is short and doesn't require scrolling, the `ScrollView` will behave like a regular layout container.

You can also achieve scrollable text programmatically by enclosing the `TextView` in a `ScrollView` programmatically in your Java or Kotlin code.

# how to bind viewmodel to layout file

To bind a `ViewModel` to a layout file in Android using the MVVM (Model-View-ViewModel) architecture, you can use the `ViewModelProvider` class to obtain an instance of the `ViewModel` and then bind it to the layout using data binding.

Here are the steps to bind a `ViewModel` to a layout file:

1. Enable data binding in your project: Open the project-level `build.gradle` file and make sure the `dataBinding` block is enabled:

```groovy
android {
    // ...
    dataBinding {
        enabled = true
    }
}
```

2. Create the layout file: Create an XML layout file where you want to bind the `ViewModel` data. For example, let's name the file `activity_main.xml`.

3. Wrap the root layout with a `<layout>` tag: In the `activity_main.xml` file, wrap the root layout with a `<layout>` tag:

```xml
<layout xmlns:android="http://schemas.android.com/apk/res/android">
    <!-- Your existing layout code here -->
</layout>
```

4. Define a variable for the `ViewModel` in the layout file: Inside the `<layout>` tag, define a variable for the `ViewModel` using the `<data>` tag:

```xml
<layout xmlns:android="http://schemas.android.com/apk/res/android">
    <data>
        <variable
            name="viewModel"
            type="com.example.MyViewModel" />
    </data>
    <!-- Your existing layout code here -->
</layout>
```

Replace `com.example.MyViewModel` with the actual class name of your `ViewModel`.

5. Bind the layout to the `ViewModel` in the activity or fragment: In your activity or fragment code, obtain an instance of the `ViewModel` using `ViewModelProvider` and then bind it to the layout using data binding:

```kotlin
// Inside your activity or fragment
private lateinit var viewModel: MyViewModel
private lateinit var binding: ActivityMainBinding

override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)

    // Obtain an instance of the ViewModel
    viewModel = ViewModelProvider(this).get(MyViewModel::class.java)

    // Inflate the layout using data binding
    binding = DataBindingUtil.setContentView(this, R.layout.activity_main)

    // Bind the ViewModel to the layout
    binding.viewModel = viewModel
    binding.lifecycleOwner = this

    // Rest of your code
}
```

Make sure to replace `com.example.MyViewModel` with the actual class name of your `ViewModel` and `ActivityMainBinding` with the generated binding class for your layout.

6. Access `ViewModel` properties in the layout: You can now access the properties and methods of the `ViewModel` in your layout using data binding expressions. For example:

```xml
<TextView
    android:layout_width="wrap_content"
    android:layout_height="wrap_content"
    android:text="@{viewModel.someProperty}" />
```

In this example, `viewModel.someProperty` refers to a property named `someProperty` in your `ViewModel`.

By following these steps, you can bind a `ViewModel` to a layout file using data binding, allowing the layout to automatically update when the `ViewModel` data changes.
