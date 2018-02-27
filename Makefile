##
## Makefile
##  
## Made by brumen
## Login   <brumenprasic>
##
## Started on  Sat Oct  8 14:19:28 2011 brumen
## Last update Sat Oct  8 14:19:28 2011 brumen
## 

compile:
#	python pricers_setup.py build_ext --inplace
	python mrds_setup.py build_ext --inplace

quartic_file:
	nvcc quartic.cu -o quartic_file

tensor_prod_omp: tensor_prod_omp.c
	gcc -fopenmp -lpython2.7 -shared -Wl,-soname,tp -fPIC -o tp.so tensor_prod_omp.c 

tensor_prod: tensor_prod.c
	gcc -lpython2.7 -lpthread -shared -Wl,-soname,tp -fPIC -o tp.so tensor_prod.c 


# backup things
backup:
	tar czvf /home/brumen/archive/mrds/mrds-`date +%F`.tar.gz /home/brumen/work/mrds

# copies the backed-up files to stick
copy_stick:
	cp /home/brumen/archive/ao/ao/public_html_archive/public_html-`date +%F`.tar.gz /media/brumen/backup/backup/
	cp /home/brumen/archive/ao/ao/ao_archive/work_ao-`date +%F`.tar.gz /media/brumen/backup/backup/
