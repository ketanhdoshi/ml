print('Interest Calculator:')

amount = float(input('Principal Amount? '))
roi = float(input('Rate of Interest? '))
yrs = int(input('Duration in Yrs? '))

total = (amount * pow(1 + (roi/100), yrs))
interest = total - amount
print('\nInterest = %0.2f' %interest)

