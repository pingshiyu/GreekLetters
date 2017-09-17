'''
GUI for a ML model, trained on the Greek Letters dataset
By Vincent Wang 2017
'''

from tkinter import Tk, Button, Canvas, Label, Toplevel, Message
from PIL import Image, ImageDraw, ImageFilter
from scipy.misc import toimage
import numpy as np
from cnn_model import Model

# dictionary (class to corresponding letter)
dict = {0: 'Alpha', 1: 'Beta', 2: 'Delta', 3: 'Epsilon', 4: 'Phi', 
        5: 'Gamma', 6: 'Eta', 7: 'Iota', 8: 'Iota', 9: 'Xi',
        10: 'Lambda', 11: 'Mu', 12: 'Nu', 13: 'Omega', 14: 'Omicron',
        15: 'Pi', 16: 'Psi', 17: 'Rho', 18: 'Sigma', 19: 'Tau', 
        20: 'Theta', 21: 'Chi', 22: 'Upsilon', 23: 'Zeta'}

def main():
    # the brain
    brain = Model('./models/2')
    
    # color constants for PIL
    white = (255, 255, 255)
    black = (0, 0, 0)
    
    root = Tk()
    
    # WINDOW PROPERTIES
    
    # makes the window translucent
    root.attributes("-alpha", 0.7)
    
    # Window name
    root.title("GlyphGuesser")
    
    # Not resizable
    root.resizable(0, 0)
    
    
    # DRAW SUITE
    
    # pointsets from befores
    priorpoints = []
    
    # images that are outputted
    backimagedata = Image.new("RGB", (250, 300), white)
    backimage = ImageDraw.Draw(backimagedata)
    
    # pointset so far
    points = []
    
    # spline to be drawn
    spline = 0
    
    # optional loop point holder
    looppoint = []
    
    tag1 = "theline"
    tag2 = "drawnpoints"    
    
    # draw a 2x2 pixel on click, and store the click location
    # this method is bound to a left-click
    def point(event):
        c.create_oval(event.x, event.y, event.x + 5, event.y +
                      5, fill="red", tags="drawnpoints")
        points.append(event.x)
        points.append(event.y)
        return points
    
    
    # start a loop node
    # bound to right-click
    def loop(event):
        try:
            if len(looppoint) == 0:
                c.create_oval(event.x, event.y, event.x + 5, event.y +
                              5, fill="blue", tags="drawnpoints")
                points.append(event.x)
                points.append(event.y)
                looppoint.append(event.x)
                looppoint.append(event.y)
                return points
            else:
                return points
        except Exception:
            pass
    
    
    def canxy(event):
        print(event.x, event.y)
    
    
    # delete all markings on the board, clear stored data so far
    # bound to d
    def delete():
        c.delete("theline", "drawnpoints")
        priorpoints[:] = []
        points[:] = []
        looppoint[:] = []
    
    # connect all points drawn so far by a line <polygonspline>
    # this method is bound to space
    def graph(event):
        try:
            if len(points) != 0:
                if len(looppoint) != 0:
                    # if there's a loop, end it
                    looper = points + looppoint
                    priorpoints.append(looper)
                    for ps in priorpoints:
                        c.create_line(ps, tags="theline", width=10)
    
                else:
                    # if there isn't...
                    noloop = points[:]
                    priorpoints.append(noloop)
                    for ps in priorpoints:
                        c.create_line(ps, tags="theline", width=10)
            # after drawing, start new set of points
            points[:] = []
            looppoint[:] = []
            # and delete the old ones
            c.delete("drawnpoints")
            return priorpoints
        except Exception:
            pass
    
    
    # def makeaguess(event):
        # todo: convert canvas to black and white thumbnail via PIL
        # convert thumbnail to 31x25 byte array, flatten
        # feed array to brain
        # print guess from brain
    
        # other todo: link brain here.
        # pickle the trained brain, call on it from somewhere
    
    
    # this method makes a polygonspline smooth
    # bound to middle-click. not visible to computer.
    def toggle(event):
        global spline
        if spline == 0:
            c.itemconfigure(tag1, smooth=1)
            spline = 1
        elif spline == 1:
            c.itemconfigure(tag1, smooth=0)
            spline = 0
        return spline
    
    
    def lookandguess():
        # create a blank canvas and draw on backimage
        backimagedata = Image.new('L', (250, 300))
        backimage = ImageDraw.Draw(backimagedata)
        
        for ps in priorpoints:
            backimage.line(ps, fill=255, width=25)
        print('prior points:', priorpoints) # debug, see content of priorpoints
        im = backimagedata.resize((45,45), Image.LINEAR)
        im = im.filter(ImageFilter.GaussianBlur(radius=2))
        
        def flatten_image(im):
            '''
            input: image object
            output: flattened, 1-d image array of shape [1, -1]
            '''
            im_arr = np.asarray(im)
            return np.reshape(im_arr, (1, -1))
        
        [result, confidence] = brain.make_prediction(flatten_image(im))
        pred_class = result
        pred_confidence = confidence
        
        # display a mesage of the predictions
        message = "I reckon: {0} ({1}, confidence: {2:.2f})".format(str(dict[pred_class]), 
                                                                         pred_class,
                                                                         pred_confidence)
        prediction_message.config(text=message)
        print(message)
        
        return priorpoints
    
    def show_tmp_message(message):
        '''
        Displays a temporary message that is destroyed after 5 seconds
        '''
        top = Toplevel()
        top.title('Welcome')
        Message(top, text=message, padx=20, pady=20).pack()
        top.after(5000, top.destroy)
    
    c = Canvas(root, bg="white", width=250, height=300)
    
    # PIL to create an empty image to draw on/export
    backimagedata = Image.new("RGB", (250, 300), white)
    backimage = ImageDraw.Draw(backimagedata)
    
    b = Button(text="Take a look!", command=lookandguess)
    b.pack()
    
    c.configure(cursor="crosshair")
    c.pack()
    
    d = Button(text="Erase!", command=delete)
    d.pack()
    
    prediction_message = Label(root, text='')
    prediction_message.pack()
    
    # canvas bindings (position sensitive)
    c.bind("<Button-1>", point)
    c.bind("<Button-3>", loop)
    
    # root bindings (not position sensitive)
    root.bind("<space>", graph)
    root.bind("s", toggle)
    
    root.mainloop()
    
if __name__ == '__main__':
    main()