import tkinter as tk
import tkinter.font as tkFont
from add_mult_func import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

disValue = 0
operator = {'+': 1, 'Squaring\nGF(2^4) x^4+x+1': 2, 'Inverse\nGF(2^4) x^4+x+1': 3, 'Ripple-Carry': 4, 'C': 5, '=': 6, 'Improved\nRipple-carry': 7, 'Carry\nLookahead': 8, 'Schoolbook': 9, 'Karatsuba': 10, 'Our\nKaratsuba': 11}
stoValue = 0
opPre = 0
count =0

### 0~9까지의 숫자를 클릭했을때
def number_click(value):
    # print('숫자 ',value)
    global disValue

    disValue = (disValue * 10) + value  # 숫자를 클릭할때마다 10의 자리씩 이동한다.
    str_value.set(disValue)  # 화면에 숫자를 나타낸다.


### C를 클릭하여 clear할때
def clear():
    global disValue, stoValue, opPre
    # 주요 변수 초기화
    stoValue = 0
    opPre = 0
    disValue = 0
    str_value.set(str(disValue))  # 화면을 지운다.
    dis1.configure(text="adder")
    dis2.configure(text="depth")
    dis3.configure(text="count")
    ax.cla()
    canvas.draw()


### + ~ = 연산자를 클릭했을때
def oprator_click(value):
    # print('명령 ', value)
    global disValue, operator, stoValue, opPre, count

    # value의 값에 따라 숫자로 연산자를 변경한다.(+는 1로, -는 2로..)
    op = operator[value]

    if op == 5:  # C (clear)
        clear()
    elif disValue == 0:  # 현재 화면에 출력된 값이 0일때
        opPre = 0
    elif opPre == 0:  # 연산자가 한번도 클릭되지 않았을때
        opPre = op  # 현재 눌린 연산자가 있으면 저장
        stoValue = disValue  # 현재까지의 숫자를 저장
        if opPre == 2:
            count, depth, ret = Squaring(stoValue)
            dis1.configure(text="4-bit Squaring")
            dis2.configure(text="depth = " + str(depth))
            dis3.configure(text="Max = %s" % bin(ret))
            disValue = ret
            ax.cla()
            ax.bar(list(count.keys()), list(count.values()), align='center')
            canvas.draw()
            disValue = 0
            stoValue = 0
            opPre = 0

        elif opPre == 3:  # inverse
            count, depth, ret = Inversion(stoValue)
            dis1.configure(text="4-bit Squaring")
            dis2.configure(text="depth = " + str(depth))
            dis3.configure(text="Max = %s" % bin(ret))
            disValue = ret
            ax.cla()
            ax.bar(list(count.keys()), list(count.values()), align='center')
            canvas.draw()
            str_value.set(str(disValue))
            disValue = 0
            stoValue = 0
            opPre = 0

        else:
            disValue = 0  # 연산자 이후의 숫자를 받기 위해 초기화
            str_value.set(str(disValue))  # 0으로 다음 숫자를 받을 준비

    elif op == 6:  # '=  결과를 계산하고 출력한다.
        if opPre == 1:  # +
            disValue = stoValue + disValue

        elif opPre == 4:  # ripple-carry
            count, depth, ret= addition4(stoValue, disValue)
            disValue = ret
            dis1.configure(text="4-bit Addition (Ripple-carry)")
            dis2.configure(text="depth = " + str(depth))
            dis3.configure(text="Max = %s" %bin(ret))

        elif opPre == 7:  # improved\nripple-carry
            count, depth, ret = CDKM(stoValue, disValue)
            disValue = ret
            dis1.configure(text="4-bit Addition (Improved Ripple-carry)")
            dis2.configure(text="depth = " + str(depth))
            dis3.configure(text="Max = %s" %bin(ret))

        elif opPre == 8:  # 'carry\nlookahead'
            count, depth, ret = QCLA(stoValue, disValue)
            disValue = ret
            dis1.configure(text="4-bit Addition (Carry Lookahead)")
            dis2.configure(text="depth = " + str(depth))
            dis3.configure(text="Max = %s" %bin(ret))

        elif opPre == 9:  # 'schoolbook'
            count, depth, ret = Schoolbook_Mul(stoValue, disValue)
            disValue = ret
            dis1.configure(text="4-bit Multiplication (Schoolbook)")
            dis2.configure(text="depth = " + str(depth))
            dis3.configure(text="Max = %s" %bin(ret))

        elif opPre == 10:  # 'karatsuba'
            count, depth, ret = Karatsuba_4bit(stoValue, disValue)
            disValue = ret
            dis1.configure(text="4-bit Multiplication (Karatsuba)")
            dis2.configure(text="depth = " + str(depth))
            dis3.configure(text="Max = %s" %bin(ret))

        elif opPre == 11:  # 'our\nkaratsuba'
            count, depth, ret = Karatsuba_Toffoli_Depth_1_4bit(stoValue, disValue)
            disValue = ret
            dis1.configure(text="4-bit Multiplication (our Karatsuba)")
            dis2.configure(text="depth = " + str(depth))
            dis3.configure(text="Max = %s" %bin(ret))

        str_value.set(str(disValue))  # 최종 결과 값을 출력한다.

        ##
        ax.cla()
        ax.bar(list(count.keys()), list(count.values()), align='center')
        canvas.draw()

        disValue = 0
        stoValue = 0
        opPre = 0
    else:
        clear()


def button_click(value):
    # print(value)
    try:
        value = int(value)  # 정수로 변환한다.
        # 정수가 아닌 경우 except가 발생하여 아래 except로 이동한다.
        number_click(value)  # 정수인 경우 number_click( )를 호출
    except:
        oprator_click(value)  # 정수가 아닌 연산자인 경우 여기로!!


win = tk.Tk()
win.title('Quantum Calculator')

str_value = tk.StringVar()
str_value.set(str(disValue))

result = tk.Entry(win, textvariable=str_value, justify='right', bg='white', fg='red') # 입력
result.grid(column=0, row=0, columnspan=4, ipadx=80, ipady=30)

fig = Figure(figsize=(8, 6), dpi=100)  # 그리프 그릴 창 생성

ax = fig.subplots() # 창에 그래프 하나 추가
ax.tick_params(axis='x', labelrotation=75)
canvas = FigureCanvasTkAgg(fig, master=win)
canvas.draw()

canvas.get_tk_widget().grid(row=0,column=4,rowspan = 10)

# textvariable 변경 요망
dis1 = tk.Label(win, text="adder", justify='left', fg='black')
dis2 = tk.Label(win, text="depth", justify='left', fg='black')
dis3 = tk.Label(win, text="count", justify='left', fg='black')
dis1.grid(column=0, row=1, columnspan=4, ipadx = 10, ipady=10)
dis2.grid(column=0, row=2, columnspan=4, ipadx = 50, ipady=10)
dis3.grid(column=0, row=3, columnspan=4, ipadx = 50, ipady=10)

calItem = [['1','2','3','4'],
           ['5', '6', '7', '8'],
           ['9', '0', 'C', '=']]

quantumItem = [['Ripple-Carry', 'Improved\nRipple-carry', 'Carry\nLookahead'],
               ['Schoolbook', 'Karatsuba', 'Our\nKaratsuba'],
               ['Squaring\nGF(2^4) x^4+x+1', 'Inverse\nGF(2^4) x^4+x+1']]

temp = ['Adder', 'Multiplier\nGF(2^4)\nx^4+x+1']

for i, items in enumerate(quantumItem):
    w, h = 10, 3
    if i<2:
        lb = tk.Label(win, text=temp[i], width = 8, height = h, bg = 'white', font=tkFont.Font(size=13, weight='bold'))
        lb.grid(column=0, row= i + 3 + 1)

    for j, item in enumerate(items):
        bt = tk.Button(win,
                       text=item,
                       width=w,
                       height=h,
                       # bg='gray',
                       fg='black',
                       command=lambda cmd=item: button_click(cmd)
                       )
        if i >= 2:
            bt.config(width = 21)
            bt.grid(column = 2*j, row = i + 3 + 1, columnspan=2)
        else:
            bt.grid(column = j + 1, row = i + 3 + 1)

for i, items in enumerate(calItem):
    for k, item in enumerate(items):
        try:
            color = int(item)
            color = 'black'
        except:
            color = 'green'
        bt = tk.Button(win,
                       text=item,
                       width=10,
                       height=5,
                       bg=color,
                       fg='black',
                       command=lambda cmd=item: button_click(cmd)
                       )
        bt.grid(column=k, row=(i + 1 + 6))
win.mainloop()

