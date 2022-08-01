#--------------------------------------
# Windows Neural Network Tool make file
#--------------------------------------

WINCC=cl -c -AL -G2sw -Ow -W3 -Zp -Tp
WINLINK=link /nod
WINLIB=lafxdwd llibcew oldnames libw commdlg ddem1
WINRC=rc -r

#

wnn.exe : wnn.obj wnn.def wnn.res
     $(WINLINK) wnn, wnn, NUL, $(WINLIB), wnn
     rc -t wnn.res

#

wnn.obj : wnn.cpp wnn.h wnnmenu.h
     $(WINCC) wnn.cpp

wnn.res : wnn.rc wnnmenu.h hello.ico hello.dlg
     $(WINRC) wnn.rc
