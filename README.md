# Image-Processing-GUI
Image Processing Tool made by me as a part of the EE610 Course

<h1 dir="auto"><a id="user-content-gui" class="anchor" aria-hidden="true" href="#gui"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>GUI</h1>
<p dir="auto"><strong>Qt creator</strong> was used to design the ui, and convert it into <em>python</em> file. The Generated <em>python</em> file was then editted to add all the button signals and a few more elements like Matplotlib's <code>FeatureCanvas</code> and <code>Qpixmap</code> for displaying the image, with an option to switch between the two with <code>use_matplotlib_backend</code> flag at the start.</p>

<p dir="auto"><code>MplCanvas</code> Class sets up the Matplotlib Canvas for Qt
<code>Ui_MainWindow</code> Class sets up the main UI Window, with <code>setupUi()</code> method adding all the elements to it including Qwidgets, Qlayouts inside the Qwidgets and adding QButtons and other smaller elements inside the Layouts as designed in <strong>Qt Creator</strong> and things that were added later manually like <code>MlpCanvas</code></p>

<p dir="auto"><code>imshow_()</code> method displays the passed image in Ui using the appropriate backend
<code>update_history()</code> method clears the QTable and updates it with latest <code>history_text</code> from its <code>self.image</code> iImage object.</p>

<p dir="auto"><code>checkout()</code> method lets the user checkout any saved image from <code>iImage.history</code> and is inspired by Git
<code>set_buttons_bindings()</code> method sets up all the button and slider binding methods that are to be called when buttons are clicked or sliders are released.</p>
