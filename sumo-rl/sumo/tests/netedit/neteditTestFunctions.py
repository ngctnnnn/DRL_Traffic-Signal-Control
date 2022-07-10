# -*- coding: utf-8 -*-
# Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.org/sumo
# Copyright (C) 2009-2022 German Aerospace Center (DLR) and others.
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# https://www.eclipse.org/legal/epl-2.0/
# This Source Code may also be made available under the following Secondary
# Licenses when the conditions for such availability set forth in the Eclipse
# Public License 2.0 are satisfied: GNU General Public License, version 2
# or later which is available at
# https://www.gnu.org/licenses/old-licenses/gpl-2.0-standalone.html
# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

# @file    neteditTestFunctions.py
# @author  Pablo Alvarez Lopez
# @date    2016-11-25

# Import libraries
from __future__ import print_function
import os
import sys
try:
    import subprocess32 as subprocess
except ImportError:
    import subprocess
import pyautogui
import time
import pyperclip
import attributesEnum as attrs  # noqa

# define delay before every operation
DELAY_KEY = 0.2
DELAY_KEY_TAB = 0.2
DELAY_MOUSE_MOVE = 0.5
DELAY_MOUSE_CLICK = 1
DELAY_QUESTION = 3
DELAY_RELOAD = 5
DELAY_QUIT_NETEDIT = 5
DELAY_UNDOREDO = 1
DELAY_SELECT = 1
DELAY_RECOMPUTE = 3
DELAY_RECOMPUTE_VOLATILE = 5
DELAY_REMOVESELECTION = 2
DELAY_CHANGEMODE = 1
DELAY_REFERENCE = 15

_NETEDIT_APP = os.environ.get("NETEDIT_BINARY", "netedit")
_TEXTTEST_SANDBOX = os.environ.get("TEXTTEST_SANDBOX", os.getcwd())
_REFERENCE_PNG = os.path.join(os.path.dirname(__file__), "reference.png")

#################################################
# interaction functions
#################################################


def typeKeyUp(key):
    """
    @brief type single key up
    """
    # Leave key up
    pyautogui.keyUp(key)
    # wait after key up
    time.sleep(DELAY_KEY)


def typeKeyDown(key):
    """
    @brief type single key down
    """
    # Leave key down
    pyautogui.keyDown(key)
    # wait after key down
    time.sleep(DELAY_KEY)


def typeEscape():
    """
    @brief type escape key
    """
    # type ESC key
    typeKey('esc')


def typeEnter():
    """
    @brief type enter key
    """
    # type enter key
    typeKey('enter')


def typeSpace():
    """
    @brief type space key
    """
    # type space key
    typeKey('space')


def typeTab():
    """
    @brief type tab key
    """
    # wait before every operation
    time.sleep(DELAY_KEY_TAB)
    # type keys
    pyautogui.hotkey('tab')


def typeBackspace():
    """
    @brief type backspace key
    """
    # wait before every operation
    time.sleep(DELAY_KEY)
    # type keys
    pyautogui.hotkey('backspace')


def typeInvertTab():
    """
    @brief type Shift + Tab keys
    """
    # wait before every operation
    time.sleep(DELAY_KEY_TAB)
    # type two keys at the same time
    pyautogui.hotkey('shift', 'tab')


def typeKey(key):
    """
    @brief type single key
    """
    # type keys
    pyautogui.hotkey(key)
    # wait before every operation
    time.sleep(DELAY_KEY)


def typeTwoKeys(key1, key2):
    """
    @brief type two keys at the same time (key1 -> key2)
    """
    # press key 1
    typeKeyDown(key1)
    # type key 2
    typeKey(key2)
    # leave key 1
    typeKeyUp(key1)


def typeThreeKeys(key1, key2, key3):
    """
    @brief type three keys at the same time (key1 -> key2 -> key3)
    """
    # press key 1
    typeKeyDown(key1)
    # type key 2 and 3
    typeTwoKeys(key2, key3)
    # leave key 1
    typeKeyUp(key1)


def translateKeys(value, layout="de"):
    """
    @brief translate keys between different keyboards
    """
    tr = {}
    if layout == "de":
        en = r"""y[];'\z/Y{}:"|Z<>?@#^&*()-_=+§"""
        de = u"""zü+öä#y-ZÜ*ÖÄ'Y;:_"§&/()=ß?´`^"""
        # join as keys and values
        tr.update(dict(zip(en, de)))
    return "".join(map(lambda x: tr.get(x, x), value))


def pasteIntoTextField(value, removePreviousContents=True, useClipboard=True, layout="de"):
    """
    @brief paste value into current text field
    """
    print(value)
    # remove previous content
    if removePreviousContents:
        typeTwoKeys('ctrl', 'a')
    if useClipboard:
        # use copy & paste (due problems with certain characters, for example '|')
        pyperclip.copy(value)
        pyautogui.hotkey('ctrl', 'v')
    else:
        pyautogui.typewrite(translateKeys(value, layout))


def leftClick(referencePosition, positionx, positiony):
    """
    @brief do left click over a position relative to referencePosition (pink square)
    """
    # obtain clicked position
    clickedPosition = [referencePosition[0] + positionx, referencePosition[1] + positiony]
    # move mouse to position
    pyautogui.moveTo(clickedPosition)
    # wait after move
    time.sleep(DELAY_MOUSE_MOVE)
    # click over position
    pyautogui.click(button='left')
    # wait after every operation
    time.sleep(DELAY_MOUSE_CLICK)
    print("TestFunctions: Clicked over position", clickedPosition[0], '-', clickedPosition[1])


def leftClickShift(referencePosition, positionx, positiony):
    """
    @brief do left click over a position relative to referencePosition (pink square) while shift key is pressed
    """
    # Leave Shift key pressed
    typeKeyDown('shift')
    # obtain clicked position
    clickedPosition = [referencePosition[0] + positionx, referencePosition[1] + positiony]
    # move mouse to position
    pyautogui.moveTo(clickedPosition)
    # wait after move
    time.sleep(DELAY_MOUSE_MOVE)
    # click over position
    pyautogui.click(button='left')
    # wait after every operation
    time.sleep(DELAY_MOUSE_CLICK)
    # show debug
    print("TestFunctions: Clicked with Shift key pressed over position", clickedPosition[0], '-', clickedPosition[1])
    # Release Shift key
    typeKeyUp('shift')


def leftClickControl(referencePosition, positionx, positiony):
    """
    @brief do left click over a position relative to referencePosition (pink square) while control key is pressed
    """
    # Leave Control key pressed
    typeKeyDown('ctrl')
    # obtain clicked position
    clickedPosition = [referencePosition[0] + positionx, referencePosition[1] + positiony]
    # move mouse to position
    pyautogui.moveTo(clickedPosition)
    # wait after move
    time.sleep(DELAY_MOUSE_MOVE)
    # click over position
    pyautogui.click(button='left')
    # wait after every operation
    time.sleep(DELAY_MOUSE_CLICK)
    # show debug
    print("TestFunctions: Clicked with Control key pressed over position", clickedPosition[0], '-', clickedPosition[1])
    # Release Control key
    typeKeyUp('ctrl')


def leftClickAltShift(referencePosition, positionx, positiony):
    """
    @brief do left click over a position relative to referencePosition (pink square) while alt key is pressed
    """
    # Leave alt key pressed
    typeKeyDown('alt')
    # Leave shift key pressed
    typeKeyDown('shift')
    # obtain clicked position
    clickedPosition = [referencePosition[0] + positionx, referencePosition[1] + positiony]
    # move mouse to position
    pyautogui.moveTo(clickedPosition)
    # wait after move
    time.sleep(DELAY_MOUSE_MOVE)
    # click over position
    pyautogui.click(button='left')
    # wait after every operation
    time.sleep(DELAY_MOUSE_CLICK)
    # show debug
    print("TestFunctions: Clicked with alt and shift key pressed over position",
          clickedPosition[0], '-', clickedPosition[1])
    # Release alt key
    typeKeyUp('alt')
    # Release shift key
    typeKeyUp('shift')


def dragDrop(referencePosition, x1, y1, x2, y2):
    """
    @brief drag and drop from position 1 to position 2
    """
    # wait before every operation
    time.sleep(DELAY_KEY)
    # obtain from and to position
    fromPosition = [referencePosition[0] + x1, referencePosition[1] + y1]
    tromPosition = [referencePosition[0] + x2, referencePosition[1] + y2]
    # move to from position
    pyautogui.moveTo(fromPosition)
    # wait before every operation
    time.sleep(DELAY_KEY)
    # drag mouse to X of 100, Y of 200 while holding down left mouse button
    pyautogui.dragTo(tromPosition[0], tromPosition[1], 1, button='left')
    # wait before every operation
    time.sleep(DELAY_KEY)

#################################################
# basic functions
#################################################


def Popen(extraParameters, debugInformation):
    """
    @brief open netedit
    """
    # set the default parameters of Netedit
    neteditCall = [_NETEDIT_APP, '--gui-testing', '--window-pos', '50,50',
                   '--window-size', '936, 697', '--no-warnings',
                   '--error-log', os.path.join(_TEXTTEST_SANDBOX, 'log.txt')]

    # check if debug output information has to be enabled
    if debugInformation:
        neteditCall += ['--gui-testing-debug']

    # check if a gui settings file has to be load
    if os.path.exists(os.path.join(_TEXTTEST_SANDBOX, "gui-settings.xml")):
        neteditCall += ['--gui-settings-file',
                        os.path.join(_TEXTTEST_SANDBOX, "gui-settings.xml")]

    # check if an existent net must be loaded
    if os.path.exists(os.path.join(_TEXTTEST_SANDBOX, "input_net.net.xml")):
        neteditCall += ['--sumo-net-file',
                        os.path.join(_TEXTTEST_SANDBOX, "input_net.net.xml")]

    # Check if additionals must be loaded
    if os.path.exists(os.path.join(_TEXTTEST_SANDBOX, "input_additionals.add.xml")):
        neteditCall += ['-a',
                        os.path.join(_TEXTTEST_SANDBOX, "input_additionals.add.xml")]

    # Check if vTypes must be loaded
    if os.path.exists(os.path.join(_TEXTTEST_SANDBOX, "input_vtypes.rou.xml")):
        neteditCall += ['-r',
                        os.path.join(_TEXTTEST_SANDBOX, "input_vtypes.rou.xml,input_routes.rou.xml")]

    elif os.path.exists(os.path.join(_TEXTTEST_SANDBOX, "input_routes.rou.xml")):
        neteditCall += ['-r',
                        os.path.join(_TEXTTEST_SANDBOX, "input_routes.rou.xml")]

    # Check if datas must be loaded
    if os.path.exists(os.path.join(_TEXTTEST_SANDBOX, "input_datas.dat.xml")):
        neteditCall += ['-d',
                        os.path.join(_TEXTTEST_SANDBOX, "input_datas.dat.xml")]

    # set output for net
    neteditCall += ['--output-file',
                    os.path.join(_TEXTTEST_SANDBOX, 'net.net.xml')]

    # set output for additionals
    neteditCall += ['--additionals-output',
                    os.path.join(_TEXTTEST_SANDBOX, "additionals.xml")]

    # set output for routes
    neteditCall += ['--demandelements-output',
                    os.path.join(_TEXTTEST_SANDBOX, "routes.xml")]

    # set output for datas
    neteditCall += ['--dataelements-output',
                    os.path.join(_TEXTTEST_SANDBOX, "datas.xml")]

    # set output for gui
    neteditCall += ['--gui-testing.setting-output',
                    os.path.join(_TEXTTEST_SANDBOX, "guisettingsoutput.xml")]

    # add extra parameters
    neteditCall += extraParameters

    # return a subprocess with Netedit
    return subprocess.Popen(neteditCall, env=os.environ, stdout=sys.stdout, stderr=sys.stderr)


def getReferenceMatch(neProcess, makeScrenshot):
    """
    @brief obtain reference referencePosition (pink square)
    """
    # show information
    print("Finding reference")
    # make a screenshot
    errorScreenshot = pyautogui.screenshot()
    try:
        # wait for reference
        time.sleep(DELAY_REFERENCE)
        # capture screen and search reference
        positionOnScreen = pyautogui.locateOnScreen(_REFERENCE_PNG, minSearchTime=3)
    except Exception as e:
        # we cannot specify the exception here because some versions of pyautogui use one and some don't
        print(e)
        positionOnScreen = None
        # make a screenshot
        errorScreenshot = pyautogui.screenshot()
    # check if pos was found
    if positionOnScreen:
        # adjust position to center
        referencePosition = (positionOnScreen[0] + 16, positionOnScreen[1] + 16)
        # break loop
        print("TestFunctions: 'reference.png' found. Position: " +
              str(referencePosition[0]) + " - " + str(referencePosition[1]))
        # check that position is consistent (due scaling)
        if referencePosition != (304, 168):
            print("TestFunctions: Position of 'reference.png' isn't consistent")
        # click over position
        pyautogui.moveTo(referencePosition)
        # wait
        time.sleep(DELAY_MOUSE_MOVE)
        # press i for inspect mode
        typeKey("i")
        # click over position (used to center view in window)
        pyautogui.click(button='left')
        # wait after every operation
        time.sleep(DELAY_MOUSE_CLICK)
        # return reference position
        return referencePosition
    # referente not found, then write screenshot
    if (makeScrenshot):
        errorScreenshot.save("errorScreenshot.png")
    # kill netedit process
    neProcess.kill()
    # print debug information
    sys.exit("TestFunctions: Killed Netedit process. 'reference.png' not found")


def setupAndStart(testRoot, extraParameters=[], debugInformation=True, makeScrenshot=True):
    """
    @brief setup and start netedit
    """
    if os.name == "posix":
        # to work around non working gtk clipboard
        pyperclip.set_clipboard("xclip")
    # Open Netedit
    neteditProcess = Popen(extraParameters, debugInformation)
    # atexit.register(quit, neteditProcess, False, False)
    # print debug information
    print("TestFunctions: Netedit opened successfully")
    # all keys up
    typeKeyUp("shift")
    typeKeyUp("control")
    typeKeyUp("alt")
    # Wait for Netedit reference
    return neteditProcess, getReferenceMatch(neteditProcess, makeScrenshot)


def supermodeNetwork():
    """
    @brief select supermode Network
    """
    typeKey('F2')


def supermodeDemand():
    """
    @brief select supermode Demand
    """
    typeKey('F3')
    # wait for output
    time.sleep(DELAY_RECOMPUTE)


def supermodeData():
    """
    @brief select supermode Data
    """
    typeKey('F4')
    # wait for output
    time.sleep(DELAY_RECOMPUTE)


def rebuildNetwork():
    """
    @brief rebuild network
    """
    typeKey('F5')
    # wait for output
    time.sleep(DELAY_RECOMPUTE)


def rebuildNetworkWithVolatileOptions(question=True):
    """
    @brief rebuild network with volatile options
    """
    typeTwoKeys('shift', 'F5')
    # confirm recompute
    if question is True:
        waitQuestion('y')
        # wait for output
        time.sleep(DELAY_RECOMPUTE_VOLATILE)
    else:
        waitQuestion('n')


def joinSelectedJunctions():
    """
    @brief join selected junctions
    """
    typeKey('F7')


def focusOnFrame():
    """
    @brief select focus on upper element of current frame
    """
    typeTwoKeys('shift', 'F12')
    time.sleep(1)


def undo(referencePosition, number, posX=0, posY=0):
    """
    @brief undo last operation
    """
    # first wait
    time.sleep(DELAY_UNDOREDO)
    # focus current frame
    focusOnFrame()
    # needed to avoid errors with undo/redo (Provisionally)
    typeKey('i')
    # click over referencePosition
    leftClick(referencePosition, posX, posY)
    for _ in range(number):
        typeTwoKeys('ctrl', 'z')
        time.sleep(DELAY_UNDOREDO)


def redo(referencePosition, number, posX=0, posY=0):
    """
    @brief undo last operation
    """
    # first wait
    time.sleep(DELAY_UNDOREDO)
    # focus current frame
    focusOnFrame()
    # needed to avoid errors with undo/redo (Provisionally)
    typeKey('i')
    # click over referencePosition
    leftClick(referencePosition, posX, posY)
    for _ in range(number):
        typeTwoKeys('ctrl', 'y')
        time.sleep(DELAY_UNDOREDO)


def setZoom(positionX, positionY, zoomLevel):
    """
    @brief set Zoom
    """
    # open edit viewport dialog
    typeTwoKeys('ctrl', 'i')
    # by default is in "load" button, then go to position X
    for _ in range(3):
        typeTab()
    # Paste position X
    pasteIntoTextField(positionX)
    # go to Y
    typeTab()
    # Paste Position Y
    pasteIntoTextField(positionY)
    # go to Z
    typeTab()
    # Paste Zoom Z
    pasteIntoTextField(zoomLevel)
    # press OK Button using shortcut
    typeTwoKeys('alt', 'o')


def waitQuestion(answer):
    """
    @brief wait question of Netedit and select a yes/no answer
    """
    # wait some second to question dialog
    time.sleep(DELAY_QUESTION)
    # Answer can be "y" or "n"
    typeTwoKeys('alt', answer)


def reload(NeteditProcess, openNetNonSavedDialog=False, saveNet=False,
           openAdditionalsNonSavedDialog=False, saveAdditionals=False,
           openDemandNonSavedDialog=False, saveDemandElements=False,
           openDataNonSavedDialog=False, saveDataElements=False):
    """
    @brief reload Netedit
    """
    # first move cursor out of magenta square
    pyautogui.moveTo(150, 200)
    # reload using hotkey
    typeTwoKeys('ctrl', 'r')
    # Check if net must be saved
    if openNetNonSavedDialog:
        # Wait some seconds
        time.sleep(DELAY_QUESTION)
        if saveNet:
            waitQuestion('s')
            # wait for log
            time.sleep(DELAY_RECOMPUTE)
        else:
            waitQuestion('q')
    # Check if additionals must be saved
    if openAdditionalsNonSavedDialog:
        # Wait some seconds
        time.sleep(DELAY_QUESTION)
        if saveAdditionals:
            waitQuestion('s')
        else:
            waitQuestion('q')
    # Check if demand elements must be saved
    if openDemandNonSavedDialog:
        # Wait some seconds
        time.sleep(DELAY_QUESTION)
        if saveDemandElements:
            waitQuestion('s')
        else:
            waitQuestion('q')
    # Check if data elements must be saved
    if openDataNonSavedDialog:
        # Wait some seconds
        time.sleep(DELAY_QUESTION)
        if saveDataElements:
            waitQuestion('s')
        else:
            waitQuestion('q')
    # Wait some seconds
    time.sleep(DELAY_RELOAD)
    # check if Netedit was crashed during reloading
    if NeteditProcess.poll() is not None:
        print("TestFunctions: Error reloading Netedit")


def quit(NeteditProcess, openNetNonSavedDialog=False, saveNet=False,
         openAdditionalsNonSavedDialog=False, saveAdditionals=False,
         openDemandNonSavedDialog=False, saveDemandElements=False,
         openDataNonSavedDialog=False, saveDataElements=False):
    """
    @brief quit Netedit
    """
    # check if Netedit is already closed
    if NeteditProcess.poll() is not None:
        # print debug information
        print("[log] TestFunctions: Netedit already closed")
    else:
        # first move cursor out of magenta square
        pyautogui.moveTo(150, 200)
        # quit using hotkey
        typeTwoKeys('ctrl', 'q')
        # Check if net must be saved
        if openNetNonSavedDialog:
            # Wait some seconds
            time.sleep(DELAY_QUESTION)
            if saveNet:
                waitQuestion('s')
                # wait for log
                time.sleep(DELAY_RECOMPUTE)
            else:
                waitQuestion('q')
        # Check if additionals must be saved
        if openAdditionalsNonSavedDialog:
            # Wait some seconds
            time.sleep(DELAY_QUESTION)
            if saveAdditionals:
                waitQuestion('s')
            else:
                waitQuestion('q')
        # Check if demand elements must be saved
        if openDemandNonSavedDialog:
            # Wait some seconds
            time.sleep(DELAY_QUESTION)
            if saveDemandElements:
                waitQuestion('s')
            else:
                waitQuestion('q')
        # Check if data elements must be saved
        if openDataNonSavedDialog:
            # Wait some seconds
            time.sleep(DELAY_QUESTION)
            if saveDataElements:
                waitQuestion('s')
            else:
                waitQuestion('q')
        # wait some seconds for netedit to quit
        if hasattr(subprocess, "TimeoutExpired"):
            try:
                NeteditProcess.wait(DELAY_QUIT_NETEDIT)
                print("TestFunctions: Netedit closed successfully")
                # all keys up
                typeKeyUp("shift")
                typeKeyUp("control")
                typeKeyUp("alt")
                # exit
                return
            except subprocess.TimeoutExpired:
                pass
        else:
            time.sleep(DELAY_QUIT_NETEDIT)
            if NeteditProcess.poll() is not None:
                print("TestFunctions: Netedit closed successfully")
                # all keys up
                typeKeyUp("shift")
                typeKeyUp("control")
                typeKeyUp("alt")
                # exit
                return
        # error closing NETEDIT then make a screenshot
        errorScreenshot = pyautogui.screenshot()
        errorScreenshot.save("errorScreenshot.png")
        # kill netedit
        NeteditProcess.kill()
        print("TestFunctions: Error closing Netedit")
        # all keys up
        typeKeyUp("shift")
        typeKeyUp("control")
        typeKeyUp("alt")
        # exit
        return


def openNetworkAs(waitTime=2):
    """
    @brief load network as
    """
    # open save network as dialog
    typeTwoKeys('ctrl', 'o')
    # jump to filename TextField
    typeTwoKeys('alt', 'f')
    pasteIntoTextField(_TEXTTEST_SANDBOX)
    typeEnter()
    pasteIntoTextField("input_net_loadedmanually.net.xml")
    typeEnter()
    # wait for saving
    time.sleep(waitTime)


def saveNetwork(referencePosition, clickOverReference=False, posX=0, posY=0):
    """
    @brief save network
    """
    # check if clickOverReference is enabled
    if clickOverReference:
        # click over reference (to avoid problem with undo-redo)
        leftClick(referencePosition, posX, posY)
    # save network using hotkey
    typeTwoKeys('ctrl', 's')
    # wait for debug (due recomputing)
    time.sleep(DELAY_RECOMPUTE)


def saveNetworkAs(waitTime=2):
    """
    @brief save network as
    """
    # open save network as dialog
    typeThreeKeys('ctrl', 'shift', 's')
    # jump to filename TextField
    typeTwoKeys('alt', 'f')
    pasteIntoTextField(_TEXTTEST_SANDBOX)
    typeEnter()
    pasteIntoTextField("net.net.xml")
    typeEnter()
    # wait for saving
    time.sleep(waitTime)
    # wait for debug
    time.sleep(DELAY_RECOMPUTE)


def forceSaveAdditionals():
    """
    @brief force save additionals
    """
    # change additional save flag using hotkey
    typeThreeKeys('ctrl', 'shift', 'u')


def forceSaveDemandElements():
    """
    @brief force save demand elements
    """
    # change demand elements save flag using hotkey
    typeThreeKeys('ctrl', 'shift', 'v')


def forceSaveDataElements():
    """
    @brief force save data elements
    """
    # change data elements save flag using hotkey
    typeThreeKeys('ctrl', 'shift', 'w')


def saveAdditionals(referencePosition, clickOverReference=False):
    """
    @brief save additionals
    """
    # check if clickOverReference is enabled
    if clickOverReference:
        # click over reference (to avoid problem with undo-redo)
        leftClick(referencePosition, 0, 0)
    # save additionals using hotkey
    typeThreeKeys('ctrl', 'shift', 'a')


def saveRoutes(referencePosition, clickOverReference=True):
    """
    @brief save routes
    """
    # check if clickOverReference is enabled
    if clickOverReference:
        # click over reference (to avoid problem with undo-redo)
        leftClick(referencePosition, 0, 0)
    # save routes using hotkey
    typeThreeKeys('ctrl', 'shift', 'd')


def saveDatas(referencePosition, clickOverReference=True, posX=0, posY=0):
    """
    @brief save datas
    """
    # check if clickOverReference is enabled
    if clickOverReference:
        # click over reference (to avoid problem with undo-redo)
        leftClick(referencePosition, posX, posY)
    # save datas using hotkey
    typeThreeKeys('ctrl', 'shift', 'b')


def fixDemandElements(solution):
    """
    @brief fix stoppingPlaces
    """
    # select bullet depending of solution
    if (solution == "saveInvalids"):
        for _ in range(3):
            typeInvertTab()
        typeSpace()
        # go back and press accept
        for _ in range(3):
            typeTab()
        typeSpace()
    elif (solution == "fixPositions"):
        for _ in range(2):
            typeInvertTab()
        typeSpace()
        # go back and press accept
        for _ in range(2):
            typeTab()
        typeSpace()
    elif (solution == "selectInvalids"):
        typeInvertTab()
        typeSpace()
        # go back and press accept
        typeTab()
        typeSpace()
    elif (solution == "activateFriendlyPos"):
        # default option, then press accept
        typeSpace()
    else:
        # press cancel
        typeTab()
        typeSpace()


def openAboutDialog(waitingTime=DELAY_QUESTION):
    """
    @brief open and close about dialog
    """
    # type F12 to open about dialog
    typeKey('F12')
    # wait before closing
    time.sleep(waitingTime)
    # press enter to close dialog (Ok must be focused)
    typeSpace()


def openConfigurationShortcut(waitTime=2):
    """
    @brief open configuration using shortcut
    """
    # open configuration dialog
    typeThreeKeys('ctrl', 'shift', 'o')
    # jump to filename TextField
    typeTwoKeys('alt', 'f')
    pasteIntoTextField(_TEXTTEST_SANDBOX)
    typeEnter()
    pasteIntoTextField("input_net.netccfg")
    typeEnter()
    # wait for loading
    time.sleep(waitTime)


def savePlainXML(waitTime=2):
    """
    @brief save configuration using shortcut
    """
    # open configuration dialog
    typeTwoKeys('ctrl', 'l')
    # jump to filename TextField
    typeTwoKeys('alt', 'f')
    pasteIntoTextField(_TEXTTEST_SANDBOX)
    typeEnter()
    pasteIntoTextField("net")
    typeEnter()
    # wait for loading
    time.sleep(waitTime)


def changeEditMode(key):
    """
    @brief Change edit mode (alt+1-9)
    """
    typeTwoKeys('alt', key)

#################################################
# Create nodes and edges
#################################################


def createEdgeMode():
    """
    @brief Change to create edge mode
    """
    typeKey('e')
    # wait for gl debug
    time.sleep(DELAY_CHANGEMODE)


def cancelEdge():
    """
    @brief Cancel current created edge (used in chain mode)
    """
    # type ESC to cancel current edge
    typeEscape()

#################################################
# Inspect mode
#################################################


def inspectMode():
    """
    @brief go to inspect mode
    """
    typeKey('i')
    # wait for gl debug
    time.sleep(DELAY_CHANGEMODE)


def modifyAttribute(attributeNumber, value, overlapped):
    """
    @brief modify attribute of type int/float/string
    """
    # focus current frame
    focusOnFrame()
    # jump to attribute depending if it's a overlapped element
    if overlapped:
        for _ in range(attributeNumber + 1 + attrs.editElements.overlapped):
            typeTab()
    else:
        for _ in range(attributeNumber + 1):
            typeTab()
    # paste the new value
    pasteIntoTextField(value)
    # type Enter to commit change
    typeEnter()


def modifyBoolAttribute(attributeNumber, overlapped):
    """
    @brief modify boolean attribute
    """
    # focus current frame
    focusOnFrame()
    # jump to attribute depending if it's a overlapped element
    if overlapped:
        for _ in range(attributeNumber + 1 + attrs.editElements.overlapped):
            typeTab()
    else:
        for _ in range(attributeNumber + 1):
            typeTab()
    # type SPACE to change value
    typeSpace()


def modifyColorAttribute(attributeNumber, color, overlapped):
    """
    @brief modify color using dialog
    """
    # focus current frame
    focusOnFrame()
    # jump to attribute depending if it's a overlapped element
    if overlapped:
        for _ in range(attributeNumber + 1 + attrs.editElements.overlapped):
            typeTab()
    else:
        for _ in range(attributeNumber + 1):
            typeTab()
    typeSpace()
    # go to list of colors TextField
    for _ in range(2):
        typeInvertTab()
    # select color
    for _ in range(1 + color):
        typeKey('down')
    # go to accept button and press it
    typeTab()
    typeSpace()


def modifyAllowDisallowValue(numTabs, overlapped):
    """
    @brief modify allow/disallow values
    """
    # open dialog
    modifyBoolAttribute(numTabs, overlapped)
    # select vtypes
    for _ in range(2):
        typeTab()
    # Change current value
    typeSpace()
    # select vtypes
    for _ in range(6):
        typeTab()
    # Change current value
    typeSpace()
    # select vtypes
    for _ in range(12):
        typeTab()
    # Change current value
    typeSpace()
    # select vtypes
    for _ in range(11):
        typeTab()
    # Change current value
    typeSpace()


def checkParameters(referencePosition, attributeNumber, overlapped):
    """
    @brief Check generic parameters
    """
    # Change generic parameters with an invalid value (dummy)
    modifyAttribute(attributeNumber, "dummyGenericParameters", overlapped)
    # Change generic parameters with an invalid value (invalid format)
    modifyAttribute(attributeNumber, "key1|key2|key3", overlapped)
    # Change generic parameters with a valid value
    modifyAttribute(attributeNumber, "key1=value1|key2=value2|key3=value3", overlapped)
    # Change generic parameters with a valid value (empty values)
    modifyAttribute(attributeNumber, "key1=|key2=|key3=", overlapped)
    # Change generic parameters with a valid value (clear parameters)
    modifyAttribute(attributeNumber, "", overlapped)
    # Change generic parameters with an valid value (duplicated keys)
    modifyAttribute(attributeNumber, "key1duplicated=value1|key1duplicated=value2|key3=value3", overlapped)
    # Change generic parameters with a valid value (duplicated values)
    modifyAttribute(attributeNumber, "key1=valueDuplicated|key2=valueDuplicated|key3=valueDuplicated", overlapped)
    # Change generic parameters with an invalid value (invalid key characters)
    modifyAttribute(attributeNumber, "keyInvalid.;%>%$$=value1|key2=value2|key3=value3", overlapped)
    # Change generic parameters with a invalid value (invalid value characters)
    modifyAttribute(attributeNumber, "key1=valueInvalid%;%$<>$$%|key2=value2|key3=value3", overlapped)
    # Change generic parameters with a valid value
    modifyAttribute(attributeNumber, "keyFinal1=value1|keyFinal2=value2|keyFinal3=value3", overlapped)
    # Check undo
    undo(referencePosition, 8)
    # Check redo
    redo(referencePosition, 8)


def checkDoubleParameters(referencePosition, attributeNumber, overlapped, posX=0, posY=0):
    """
    @brief Check generic parameters
    """
    # Change generic parameters with an invalid value (dummy)
    modifyAttribute(attributeNumber, "dummyGenericParameters", overlapped)
    # Change generic parameters with an invalid value (invalid format)
    modifyAttribute(attributeNumber, "key1|key2|key3", overlapped)
    # Change generic parameters with a valid value
    modifyAttribute(attributeNumber, "key1=1|key2=2|key3=3", overlapped)
    # Change generic parameters with a valid value (empty values)
    modifyAttribute(attributeNumber, "key1=|key2=|key3=", overlapped)
    # Change generic parameters with a valid value (clear parameters)
    modifyAttribute(attributeNumber, "", overlapped)
    # Change generic parameters with an valid value (duplicated keys)
    modifyAttribute(attributeNumber, "key1duplicated=1|key1duplicated=2|key3=3", overlapped)
    # Change generic parameters with a valid value (duplicated values)
    modifyAttribute(attributeNumber, "key1=valueDuplicated|key2=valueDuplicated|key3=valueDuplicated", overlapped)
    # Change generic parameters with an invalid value (invalid key characters)
    modifyAttribute(attributeNumber, "keyInvalid.;%>%$$=1|key2=2|key3=3", overlapped)
    # Change generic parameters with a invalid value (invalid value characters)
    modifyAttribute(attributeNumber, "key1=valueInvalid%;%$<>$$%|key2=2|key3=3", overlapped)
    # Change generic parameters with a valid value
    modifyAttribute(attributeNumber, "keyFinal1=1|keyFinal2=2|keyFinal3=3", overlapped)
    # Check undo (including load/creation)
    undo(referencePosition, 8)
    # Check redo
    redo(referencePosition, 8)

#################################################
# Move mode
#################################################


def moveMode():
    """
    @brief set move mode
    """
    typeKey('m')
    # wait for gl debug
    time.sleep(DELAY_CHANGEMODE)


def moveElement(referencePosition, startX, startY, endX, endY):
    """
    @brief move element
    """
    # move element
    dragDrop(referencePosition, startX, startY, endX, endY)

#################################################
# crossings
#################################################


def crossingMode():
    """
    @brief Change to crossing mode
    """
    typeKey('r')
    # wait for gl debug
    time.sleep(DELAY_CHANGEMODE)


def createCrossing(hasTLS):
    """
    @brief create crossing
    """
    # focus current frame
    focusOnFrame()
    # jump to create crossing button depending of hasTLS
    if hasTLS:
        for _ in range(attrs.TLS.create.TLS):
            typeTab()
    else:
        for _ in range(attrs.TLS.create.noTLS):
            typeTab()
    # type space to create crossing
    typeSpace()


def modifyCrossingDefaultValue(numtabs, value):
    """
    @brief change default int/real/string crossing default value
    """
    # focus current frame
    focusOnFrame()
    # jump to value
    for _ in range(numtabs + attrs.crossing.firstField):
        typeTab()
    # paste the new value
    pasteIntoTextField(value)
    # type enter to save change
    typeEnter()


def modifyCrossingDefaultBoolValue(numtabs):
    """
    @brief change default boolean crossing default value
    """
    # focus current frame
    focusOnFrame()
    # jump to value
    for _ in range(numtabs + attrs.crossing.firstField):
        typeTab()
    # type space to change value
    typeSpace()


def crossingClearEdges(useSelectedEdges=False, thereIsSelectedEdges=False):
    """
    @brief clear crossing
    """
    # focus current frame
    focusOnFrame()
    if(useSelectedEdges and thereIsSelectedEdges):
        # jump to clear button
        for _ in range(attrs.crossing.clearEdgesSelected):
            typeTab()
    else:
        # jump to clear button
        for _ in range(attrs.crossing.clearEdges):
            typeTab()
    # type space to activate button
    typeSpace()


def crossingInvertEdges(useSelectedEdges=False, thereIsSelectedEdges=False):
    """
    @brief invert crossing
    """
    # focus current frame
    focusOnFrame()
    if(useSelectedEdges and thereIsSelectedEdges):
        # jump to clear button
        for _ in range(attrs.crossing.clearEdgesSelected):
            typeTab()
    else:
        # jump to clear button
        for _ in range(attrs.crossing.clearEdges):
            typeTab()
    # type space to activate button
    typeSpace()

#################################################
# Connection mode
#################################################


def connectionMode():
    """
    @brief Change to connection mode
    """
    typeKey('c')
    # wait for gl debug
    time.sleep(DELAY_CHANGEMODE)


def createConnection(referencePosition, fromLanePositionX, fromLanePositionY,
                     toLanePositionX, toLanePositionY, mode=""):
    """
    @brief create connection
    """
    # check if connection has to be created in certain mode
    if mode == "conflict":
        typeKeyDown('ctrl')
    elif mode == "yield":
        typeKeyDown('shift')
    # select first lane
    leftClick(referencePosition, fromLanePositionX, fromLanePositionY)
    # select another lane for create a connection
    leftClick(referencePosition, toLanePositionX, toLanePositionY)
    # check if connection has to be created in certain mode
    if mode == "conflict":
        typeKeyUp('ctrl')
    elif mode == "yield":
        typeKeyUp('shift')


def saveConnectionEdit():
    """
    @brief Change to crossing mode
    """
    # focus current frame
    focusOnFrame()
    # go to cancel button
    for _ in range(attrs.connection.saveConnections):
        typeTab()
    # type space to press button
    typeSpace()
    # wait for gl debug
    time.sleep(DELAY_SELECT)

#################################################
# additionals
#################################################


def additionalMode():
    """
    @brief change to additional mode
    """
    typeKey('a')
    # wait for gl debug
    time.sleep(DELAY_CHANGEMODE)


def changeElement(element):
    """
    @brief change element (Additional, shape, vehicle...)
    """
    # focus current frame
    focusOnFrame()
    # go to first editable element of frame
    for _ in range(attrs.additionals.changeElement):
        typeTab()
    # paste the new value
    pasteIntoTextField(element)
    # type enter to save change
    typeEnter()


def changeDefaultValue(numTabs, value):
    """
    @brief modify default int/double/string value of an additional, shape, vehicle...
    """
    # focus current frame
    focusOnFrame()
    # go to value TextField
    for _ in range(numTabs):
        typeTab()
    # paste new value
    pasteIntoTextField(value)
    # type enter to save new value
    typeEnter()


def changeDefaultBoolValue(numTabs):

    # focus current frame
    focusOnFrame()
    # place cursor in check Box position
    for _ in range(numTabs):
        typeTab()
    # Change current value
    typeSpace()


def changeDefaultAllowDisallowValue(numTabs):
    """
    @brief modify allow/disallow values
    """
    # open dialog
    changeDefaultBoolValue(numTabs)
    # select vtypes
    for _ in range(2):
        typeTab()
    # Change current value
    typeSpace()
    # select vtypes
    for _ in range(6):
        typeTab()
    # Change current value
    typeSpace()
    # select vtypes
    for _ in range(12):
        typeTab()
    # Change current value
    typeSpace()
    # select vtypes
    for _ in range(11):
        typeTab()
    # Change current value
    typeSpace()


def selectAdditionalChild(numTabs, childNumber):
    """
    @brief select child of additional
    """
    # focus current frame
    focusOnFrame()
    # place cursor in the list of childs
    for _ in range(numTabs + 1):
        typeTab()
    # select child
    for _ in range(childNumber):
        typeKey('down')
    typeSpace()
    # use TAB to select additional child
    typeTab()


def fixStoppingPlace(solution):
    """
    @brief fix stoppingPlaces
    """
    # wait some second to question dialog
    time.sleep(DELAY_QUESTION)
    # select bullet depending of solution
    if (solution == "saveInvalids"):
        for _ in range(3):
            typeInvertTab()
        typeSpace()
        # go back and press accept
        for _ in range(3):
            typeTab()
        typeSpace()
    elif (solution == "fixPositions"):
        for _ in range(2):
            typeInvertTab()
        typeSpace()
        # go back and press accept
        for _ in range(2):
            typeTab()
        typeSpace()
    elif (solution == "selectInvalids"):
        typeInvertTab()
        typeSpace()
        # go back and press accept
        typeTab()
        typeSpace()
    elif (solution == "activateFriendlyPos"):
        # default option, then press accept
        typeSpace()
    else:
        # press cancel
        typeTab()
        typeSpace()

#################################################
# demand elements
#################################################


def routeMode():
    """
    @brief change to route mode
    """
    typeKey('r')
    # wait for gl debug
    time.sleep(DELAY_CHANGEMODE)


def changeRouteMode(value):
    """
    @brief change route mode
    """
    # focus current frame
    focusOnFrame()
    # jump to route mode
    for _ in range(2):
        typeTab()
    # paste the new value
    pasteIntoTextField(value)
    # type enter to save change
    typeEnter()


def changeRouteVClass(value):
    """
    @brief change vClass mode
    """
    # focus current frame
    focusOnFrame()
    # jump to vClass
    for _ in range(4):
        typeTab()
    # paste the new value
    pasteIntoTextField(value)
    # type enter to save change
    typeEnter()


def fixDemandElement(value):
    """
    @brief fix demand element
    """
    # focus current frame
    focusOnFrame()
    # jump to option
    for _ in range(value):
        typeInvertTab()
    # type space to select
    typeSpace()
    # accept
    typeTwoKeys('alt', 'a')

#################################################
# person elements
#################################################


def personMode():
    """
    @brief change to person mode
    """
    typeKey('p')
    # wait for gl debug
    time.sleep(DELAY_CHANGEMODE)


def changePersonMode(value):
    """
    @brief change person mode
    """
    # focus current frame
    focusOnFrame()
    # jump to person mode
    typeTab()
    # paste the new value
    pasteIntoTextField(value)
    # type enter to save change
    typeEnter()


def changePersonVClass(value):
    """
    @brief change vClass mode
    """
    # focus current frame
    focusOnFrame()
    # jump to vClass
    for _ in range(3):
        typeTab()
    # paste the new value
    pasteIntoTextField(value)
    # type enter to save change
    typeEnter()


def changePersonPlan(personPlan, flow):
    """
    @brief change personPlan
    """
    # focus current frame
    focusOnFrame()
    # jump to person plan
    if (flow):
        for _ in range(23):
            typeTab()
    else:
        for _ in range(16):
            typeTab()
    # paste the new personPlan
    pasteIntoTextField(personPlan)
    # type enter to save change
    typeEnter()


def changePersonFlowPlan(personFlowPlan):
    """
    @brief change personFlowPlan
    """
    # focus current frame
    focusOnFrame()
    # jump to personFlow plan
    for _ in range(23):
        typeTab()
    # paste the new personFlowPlan
    pasteIntoTextField(personFlowPlan)
    # type enter to save change
    typeEnter()

#################################################
# container elements
#################################################


def containerMode():
    """
    @brief change to container mode
    """
    typeKey('g')
    # wait for gl debug
    time.sleep(DELAY_CHANGEMODE)


def changeContainerMode(value):
    """
    @brief change container mode
    """
    # focus current frame
    focusOnFrame()
    # jump to container mode
    typeTab()
    # paste the new value
    pasteIntoTextField(value)
    # type enter to save change
    typeEnter()


def changeContainerVClass(value):
    """
    @brief change vClass mode
    """
    # focus current frame
    focusOnFrame()
    # jump to vClass
    for _ in range(3):
        typeTab()
    # paste the new value
    pasteIntoTextField(value)
    # type enter to save change
    typeEnter()


def changeContainerPlan(containerPlan, flow):
    """
    @brief change containerPlan
    """
    # focus current frame
    focusOnFrame()
    # jump to container plan
    if (flow):
        for _ in range(22):
            typeTab()
    else:
        for _ in range(15):
            typeTab()
    # paste the new containerPlan
    pasteIntoTextField(containerPlan)
    # type enter to save change
    typeEnter()


def changeContainerFlowPlan(containerFlowPlan):
    """
    @brief change containerFlowPlan
    """
    # focus current frame
    focusOnFrame()
    # jump to containerFlow plan
    for _ in range(23):
        typeTab()
    # paste the new containerFlowPlan
    pasteIntoTextField(containerFlowPlan)
    # type enter to save change
    typeEnter()

#################################################
# stop elements
#################################################


def stopMode():
    """
    @brief change to person mode
    """
    typeKey('a')


def changeStopParent(stopParent):
    """
    @brief change stop parent
    """
    # focus current frame
    focusOnFrame()
    for _ in range(2):
        typeTab()
    # paste the new stop parent
    pasteIntoTextField(stopParent)
    # type enter to save change
    typeEnter()


def changeStopType(stopType):
    """
    @brief change stop type
    """
    # focus current frame
    focusOnFrame()
    # jump to stop type
    for _ in range(5):
        typeTab()
    # paste the new personPlan
    pasteIntoTextField(stopType)
    # type enter to save change
    typeEnter()

#################################################
# vehicle elements
#################################################


def vehicleMode():
    """
    @brief change to vehicle mode
    """
    typeKey('v')
    # wait for gl debug
    time.sleep(DELAY_CHANGEMODE)

#################################################
# vType elements
#################################################


def typeMode():
    """
    @brief change to type mode
    """
    typeKey('t')
    # wait for gl debug
    time.sleep(DELAY_CHANGEMODE)


def createVType():
    """
    @brief create vType
    """
    # focus current frame
    focusOnFrame()
    # jump to stop type
    for _ in range(attrs.type.buttons.create):
        typeTab()
    # type space
    typeSpace()


def deleteVType():
    """
    @brief delete vType
    """
    # focus current frame
    focusOnFrame()
    # jump to stop type
    for _ in range(attrs.type.buttons.delete):
        typeTab()
    # type space
    typeSpace()


def copyVType():
    """
    @brief copy vType
    """
    # focus current frame
    focusOnFrame()
    # jump to stop type
    for _ in range(attrs.type.buttons.copy):
        typeTab()
    # type space
    typeSpace()


def openVTypeDialog():
    """
    @brief create vType
    """
    # focus current frame
    focusOnFrame()
    # jump to stop type
    for _ in range(attrs.type.buttons.dialog):
        typeTab()
    # type space
    typeSpace()
    # wait some second to question dialog
    time.sleep(DELAY_QUESTION)


def closeVTypeDialog():
    """
    @brief close vType dialog saving elements
    """
    typeTwoKeys('alt', 'a')


def modifyVTypeAttribute(attributeNumber, value):
    """
    @brief modify VType attribute of type int/float/string
    """
    # focus dialog
    typeTwoKeys('alt', 'f')
    # jump to attribute
    for _ in range(attributeNumber):
        typeTab()
    # paste the new value
    pasteIntoTextField(value)
    # type Enter to commit change
    typeEnter()

#################################################
# delete
#################################################


def deleteMode():
    """
    @brief Change to delete mode
    """
    typeKey('d')
    # wait for gl debug
    time.sleep(DELAY_CHANGEMODE)


def deleteUsingSuprKey():
    """
    @brief delete using SUPR key
    """
    typeKey('del')
    # wait for GL Debug
    time.sleep(DELAY_REMOVESELECTION)


def changeRemoveOnlyGeometryPoint(referencePosition):
    """
    @brief Enable or disable 'Remove only geometry point'
    """
    # select delete mode again to set mode
    deleteMode()
    # jump to checkbox
    typeTab()
    # type SPACE to change value
    typeSpace()


def changeProtectAdditionalElements(referencePosition):
    """
    @brief Enable or disable 'automatically delete Additionals'
    """
    # select delete mode again to set mode
    deleteMode()
    # jump to checkbox
    for _ in range(4):
        typeTab()
    # type SPACE to change value
    typeSpace()


def changeProtectTAZElements(referencePosition):
    """
    @brief Enable or disable 'protect TAZ elements'
    """
    # select delete mode again to set mode
    deleteMode()
    # jump to checkbox
    for _ in range(5):
        typeTab()
    # type SPACE to change value
    typeSpace()


def changeProtectDemandElements(referencePosition):
    """
    @brief Enable or disable 'protect demand elements'
    """
    # select delete mode again to set mode
    deleteMode()
    # jump to checkbox
    for _ in range(6):
        typeTab()
    # type SPACE to change value
    typeSpace()


def changeProtectDataElements(referencePosition):
    """
    @brief Enable or disable 'protect data elements'
    """
    # select delete mode again to set mode
    deleteMode()
    # jump to checkbox
    for _ in range(7):
        typeTab()
    # type SPACE to change value
    typeSpace()


def waitDeleteWarning():
    """
    @brief close warning about automatically delete additionals
    """
    # wait 0.5 second to question dialog
    time.sleep(DELAY_QUESTION)
    # press enter to close dialog
    typeEnter()

#################################################
# select mode
#################################################


def selectMode():
    """
    @brief Change to select mode
    """
    typeKey('s')
    # wait for gl debug
    time.sleep(DELAY_CHANGEMODE)


def abortSelection():
    """
    @brief abort current selection
    """
    # type ESC to abort current selection
    typeEscape()


def lockSelection(glType):
    """
    @brief lock selection by glType
    """
    # focus current frame
    focusOnFrame()
    # move mouse
    pyautogui.moveTo(550, 200)
    # open Lock menu
    typeTwoKeys('alt', 'o')
    # go to selected glType
    for _ in range(glType):
        typeKey("down")
    # type enter to save change
    typeSpace()


def selectDefault():
    """
    @brief select elements with default frame values
    """
    # focus current frame
    focusOnFrame()
    for _ in range(15):
        typeTab()
    # type enter to select it
    typeEnter()
    # wait for gl debug
    time.sleep(DELAY_SELECT)


def saveSelection():
    """
    @brief save selection
    """
    focusOnFrame()
    # jump to save
    for _ in range(22):
        typeTab()
    typeSpace()
    # jump to filename TextField
    typeTwoKeys('alt', 'f')
    filename = os.path.join(_TEXTTEST_SANDBOX, "selection.txt")
    pasteIntoTextField(filename)
    typeEnter()


def loadSelection():
    """
    @brief save selection
    """
    focusOnFrame()
    # jump to save
    for _ in range(25):
        typeTab()
    typeSpace()
    # jump to filename TextField
    typeTwoKeys('alt', 'f')
    filename = os.path.join(_TEXTTEST_SANDBOX, "selection.txt")
    pasteIntoTextField(filename)
    typeEnter()
    # wait for gl debug
    time.sleep(DELAY_SELECT)


def selectItems(elementClass, elementType, attribute, value):
    """
    @brief select items
    """
    # focus current frame
    focusOnFrame()
    # jump to elementClass
    for _ in range(8):
        typeTab()
    # paste the new elementClass
    pasteIntoTextField(elementClass)
    # jump to element
    for _ in range(3):
        typeTab()
    # paste the new elementType
    pasteIntoTextField(elementType)
    # jump to attribute
    for _ in range(2):
        typeTab()
    # paste the new attribute
    pasteIntoTextField(attribute)
    # jump to value
    for _ in range(2):
        typeTab()
    # paste the new value
    pasteIntoTextField(value)
    # type enter to select it
    typeEnter()
    # wait for gl debug
    time.sleep(DELAY_SELECT)


def deleteSelectedItems():
    """
    @brief delete selected items
    """
    typeKey('del')
    # wait for gl debug
    time.sleep(DELAY_SELECT)


def modificationModeAdd():
    """
    @brief set modification mode "add"
    """
    # focus current frame
    focusOnFrame()
    # jump to mode "add"
    for _ in range(3):
        typeTab()
    # select it
    typeSpace()


def modificationModeRemove():
    """
    @brief set modification mode "remove"
    """
    # focus current frame
    focusOnFrame()
    # jump to mode "remove"
    for _ in range(4):
        typeTab()
    # select it
    typeSpace()


def modificationModeKeep():
    """
    @brief set modification mode "keep"
    """
    # focus current frame
    focusOnFrame()
    # jump to mode "keep"
    for _ in range(5):
        typeTab()
    # select it
    typeSpace()


def modificationModeReplace():
    """
    @brief set modification mode "replace"
    """
    # focus current frame
    focusOnFrame()
    # jump to mode "replace"
    for _ in range(6):
        typeTab()
    # select it
    typeSpace()


def selectionRectangle(referencePosition, startX, startY, endX, endY):
    """
    @brief select using an rectangle
    """
    # Leave Shift key pressedX
    typeKeyDown('shift')
    # move element
    dragDrop(referencePosition, startX, startY, endX, endY)
    # wait after key up
    time.sleep(DELAY_KEY)
    # Release Shift key
    typeKeyUp('shift')
    # wait for gl debug
    time.sleep(DELAY_SELECT)


def selectionApply():
    """
    @brief apply selection
    """
    # focus current frame
    focusOnFrame()
    for _ in range(16):
        typeTab()
    # type space to select clear option
    typeSpace()
    # wait for gl debug
    time.sleep(DELAY_SELECT)


def selectionClear():
    """
    @brief clear selection
    """
    # focus current frame
    focusOnFrame()
    for _ in range(21):
        typeTab()
    # type space to select clear option
    typeSpace()
    # wait for gl debug
    time.sleep(DELAY_SELECT)


def selectionInvert():
    """
    @brief invert selection
    """
    # focus current frame
    focusOnFrame()
    for _ in range(24):
        typeTab()
    # type space to select invert operation
    typeSpace()
    # wait for gl debug
    time.sleep(DELAY_SELECT)


def selectionInvertData():
    """
    @brief invert selection
    """
    # focus current frame
    focusOnFrame()
    for _ in range(27):
        typeTab()
    # type space to select invert operation
    typeSpace()
    # wait for gl debug
    time.sleep(DELAY_SELECT)


#################################################
# traffic light
#################################################

def selectTLSMode():
    """
    @brief Change to traffic light mode
    """
    typeKey('t')
    # wait for gl debug
    time.sleep(DELAY_CHANGEMODE)


def createTLS():
    """
    @brief Create TLS in the current selected Junction
    """
    # focus current frame
    focusOnFrame()
    # type tab 2 times to jump to create TLS button
    for _ in range(4):
        typeTab()
    # create TLS
    typeSpace()

#################################################
# shapes
#################################################


def shapeMode():
    """
    @brief change to shape mode
    """
    typeKey('p')
    # wait for gl debug
    time.sleep(DELAY_CHANGEMODE)


def createSquaredPoly(referencePosition, positionx, positiony, size, close):
    """
    @brief Create squared Polygon in position with a certain size
    """
    # focus current frame
    focusOnFrame()
    # start draw
    typeEnter()
    # create polygon
    leftClick(referencePosition, positionx, positiony)
    leftClick(referencePosition, positionx, positiony - (size / 2))
    leftClick(referencePosition, positionx - (size / 2), positiony - (size / 2))
    leftClick(referencePosition, positionx - (size / 2), positiony)
    # check if polygon has to be closed
    if (close is True):
        leftClick(referencePosition, positionx, positiony)
    # finish draw
    typeEnter()


def createRectangledPoly(referencePosition, positionx, positiony, sizex, sizey, close):
    """
    @brief Create rectangle Polygon in position with a certain size
    """
    # focus current frame
    focusOnFrame()
    # start draw
    typeEnter()
    # create polygon
    leftClick(referencePosition, positionx, positiony)
    leftClick(referencePosition, positionx, positiony - (sizey / 2))
    leftClick(referencePosition, positionx - (sizex / 2), positiony - (sizey / 2))
    leftClick(referencePosition, positionx - (sizex / 2), positiony)
    # check if polygon has to be closed
    if (close is True):
        leftClick(referencePosition, positionx, positiony)
    # finish draw
    typeEnter()


def createLinePoly(referencePosition, positionx, positiony, sizex, sizey, close):
    """
    @brief Create line Polygon in position with a certain size
    """
    # focus current frame
    focusOnFrame()
    # start draw
    typeEnter()
    # create polygon
    leftClick(referencePosition, positionx, positiony)
    leftClick(referencePosition, positionx - (sizex / 2), positiony - (sizey / 2))
    # check if polygon has to be closed
    if (close is True):
        leftClick(referencePosition, positionx, positiony)
    # finish draw
    typeEnter()


def changeColorUsingDialog(numTabs, color):
    """
    @brief modify default color using dialog
    """
    # focus current frame
    focusOnFrame()
    # go to length TextField
    for _ in range(numTabs):
        typeTab()
    typeSpace()
    # go to list of colors TextField
    for _ in range(2):
        typeInvertTab()
    # select color
    for _ in range(1 + color):
        typeKey('down')
    # go to accept button and press it
    typeTab()
    typeSpace()


def createGEOPOI():
    """
    @brief create GEO POI
    """
    # focus current frame
    focusOnFrame()
    # place cursor in create GEO POI
    for _ in range(20):
        typeTab()
    # create geoPOI
    typeSpace()


def GEOPOILonLat():
    """
    @brief change GEO POI format as Lon Lat
    """
    # focus current frame
    focusOnFrame()
    # place cursor in lon-lat
    for _ in range(16):
        typeTab()
    # Change current value
    typeSpace()


def GEOPOILatLon():
    """
    @brief change GEO POI format as Lat Lon
    """
    # focus current frame
    focusOnFrame()
    # place cursor in lat-lon
    for _ in range(17):
        typeTab()
    # Change current value
    typeSpace()


#################################################
# TAZs
#################################################

def TAZMode():
    """
    @brief change to TAZ mode
    """
    typeKey('z')
    # wait for gl debug
    time.sleep(DELAY_CHANGEMODE)


def createSquaredTAZ(referencePosition, positionx, positiony, size, close):
    """
    @brief Create squared TAZ in position with a certain size
    """
    # focus current frame
    focusOnFrame()
    # start draw
    typeEnter()
    # create TAZ
    leftClick(referencePosition, positionx, positiony)
    leftClick(referencePosition, positionx, positiony - (size / 2))
    leftClick(referencePosition, positionx - (size / 2), positiony - (size / 2))
    leftClick(referencePosition, positionx - (size / 2), positiony)
    # check if TAZ has to be closed
    if (close is True):
        leftClick(referencePosition, positionx, positiony)
    # finish draw
    typeEnter()


def createLineTAZ(referencePosition, positionx, positiony, sizex, sizey, close):
    """
    @brief Create line TAZ in position with a certain size
    """
    # focus current frame
    focusOnFrame()
    # start draw
    typeEnter()
    # create TAZ
    leftClick(referencePosition, positionx, positiony)
    leftClick(referencePosition, positionx - (sizex / 2), positiony - (sizey / 2))
    # check if TAZ has to be closed
    if (close is True):
        leftClick(referencePosition, positionx, positiony)
    # finish draw
    typeEnter()

#################################################
# datas
#################################################


def edgeData():
    """
    @brief change to edgeData mode
    """
    typeKey('e')
    # wait for gl debug
    time.sleep(DELAY_CHANGEMODE)


def edgeRelData():
    """
    @brief change to edgeRelData mode
    """
    typeKey('r')
    # wait for gl debug
    time.sleep(DELAY_CHANGEMODE)


def TAZRelData():
    """
    @brief change to TAZRelData mode
    """
    typeKey('z')
    # wait for gl debug
    time.sleep(DELAY_CHANGEMODE)


def createDataSet(dataSetID="newDataSet"):
    """
    @brief create dataSet
    """
    # focus current frame
    focusOnFrame()
    # go to create new dataSet
    for _ in range(2):
        typeTab()
    # enable create dataSet
    typeSpace()
    # go to create new dataSet
    typeTab()
    # create new ID
    pasteIntoTextField(dataSetID)
    # go to create new dataSet button
    typeTab()
    # create dataSet
    typeSpace()


def createDataInterval(begin="0", end="3600"):
    """
    @brief create dataInterval
    """
    # focus current frame
    focusOnFrame()
    # go to create new dataInterval
    for _ in range(5):
        typeTab()
    typeTab()
    # enable create dataInterval
    typeSpace()
    # go to create new dataInterval begin
    typeTab()
    # set begin
    pasteIntoTextField(begin)
    # go to end
    typeTab()
    # set end
    pasteIntoTextField(end)
    # go to create new dataSet button
    typeTab()
    # create dataSet
    typeSpace()

#################################################
# Contextual menu
#################################################


def contextualMenuOperation(referencePosition, positionx, positiony, operation, suboperation1, suboperation2=0):
    # obtain clicked position
    clickedPosition = [referencePosition[0] + positionx, referencePosition[1] + positiony]
    # click relative to offset
    pyautogui.rightClick(clickedPosition)
    # place cursor over first operation
    for _ in range(operation):
        # wait before every down
        time.sleep(DELAY_KEY_TAB)
        # type down keys
        pyautogui.hotkey('down')
    if suboperation1 > 0:
        # type right key for the second menu
        typeSpace()
        # place cursor over second operation
        for _ in range(suboperation1):
            # wait before every down
            time.sleep(DELAY_KEY_TAB)
            # type down keys
            pyautogui.hotkey('down')
    if suboperation2 > 0:
        # type right key for the third menu
        typeSpace()
        # place cursor over third operation
        for _ in range(suboperation2):
            # wait before every down
            time.sleep(DELAY_KEY_TAB)
            # type down keys
            pyautogui.hotkey('down')
    # select current operation
    typeSpace()
