# Przydatne skrypty i komendy

## Jak rozpocząć pracę

1. Zaloguj się na klaster Rysy.
2. Pobierz kontener `nvidia_container.sif`.
3. Pobierz zbiór danych CIFAR-10.
4. Uruchom zadanie na wybranym węźle lub uruchom batch z treningiem.
5. Uruchom środowisko Singularity.
6. Po rozpoczęciu treningu na ekranie pojawi się identyfikator zadania (`id_task`). Przejdź do folderu `output` i sprawdź pliki `id_task.err` oraz `id_task.out`.  
   W pliku `.err` można zignorować ostrzeżenia typu `more than 50 (471) bind mounts` oraz te związane z równoległością.  
   W pliku `.out` pojawią się informacje o trenowanym modelu. Upewnij się, że używana jest właściwa funkcja straty (`training.loss`) oraz odpowiednia klasa modelu. Na końcu pliku znajdziesz ścieżkę do folderu, w którym zapisywane są wagi modelu (`run_dir`).

7. Jeśli trening przebiega bez błędów, w podanym folderze będą pojawiać się snapshoty wag modelu. Im wyższy numer snapshotu, tym bardziej wytrenowany model.

## Przydatne skrypty

Na początku nadaj uprawnienia do wykonywania skryptów:
```bash
chmod u+x *.sh
```

### Uruchamianie zadania na wybranym węźle

Aby uruchomić obliczenia na węźle, użyj skryptu `./run_task.sh`, który przyjmuje dwa argumenty: liczbę kart GPU oraz czas trwania zadania. Przykład wywołania:
```bash
./run_task.sh 2 4
```
Jeśli nie podasz argumentów, domyślnie zostanie użyta jedna karta GPU oraz czas trwania 1 godzina. Maksymalna liczba GPU to 8, a maksymalny czas sesji to 48 godzin.  
Więcej informacji: [Komputery w ICM](https://kdm.icm.edu.pl/Zasoby/komputery_w_icm.pl/)

### Uruchamianie środowiska Singularity

Aby uruchomić środowisko Singularity, użyj:
```bash
./run_singularity.sh
```
Domyślna nazwa kontenera to `nvidia-container.sif`. W przypadku innej nazwy należy zmodyfikować skrypt.

### Trening modelu

Aby rozpocząć trening, użyj polecenia:
```bash
sbatch train.batch
```
W pliku `train.batch` wywoływany jest skrypt `train.sh` z odpowiednimi parametrami. Zwróć uwagę na ustawienia czasu (`--time`) oraz liczbę kart graficznych (`--gres=gpu:4`).  
Pełna dokumentacja parametrów: [Slurm w ICM](https://kdm.icm.edu.pl/Tutorials/HPC-intro/slurm_intro/).

Przykładowe wywołanie programu w `train.sh`:
Zakładamy, że baza danych nazywa się `cifar10-32x32.zip` i znajduje się w tym samym folderze co `train.py`. Możesz ustawić parametry takie jak `batch-gpu` oraz wybrać model (np. `discrete-ddp` to nazwa nowego modelu, który został dodany).

## Inne przydatne komendy

### Pobieranie danych z Rysów

Możesz użyć SFTP:
```bash
sftp nazwa_użytkownika@hpc.icm.edu.pl
get nazwa_pliku
```

### Sprawdzanie dostępności urządzeń

```bash
squeue
```

### Kopiowanie plików z Rysów na HPC/ICM

```bash
scp nazwa_pliku /lu/tetyda/home/ms440009/
```

### Problemy z kluczem publicznym GitHub

Sprawdź konfigurację kluczy:
```bash
ssh -T git@github.com
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/licencjat
```
(Zamiast `licencjat` wpisz nazwę swojego klucza prywatnego)

### Jak dodać własny model?

Aby dodać własny model do projektu, wykonaj następujące kroki:

1. **Dodaj nową flagę w `train.py`**  
   Zdefiniuj nową opcję (np. `--precond`) umożliwiającą wybór Twojego modelu podczas uruchamiania treningu.

2. **Zaimplementuj własną funkcję straty w `loss.py`**  
   Dodaj klasę lub funkcję odpowiadającą za obliczanie straty dla Twojego modelu.

3. **Zdefiniuj architekturę sieci w `network.py`**  
   Utwórz klasę opisującą strukturę Twojego modelu.

4. **Zaktualizuj skrypt `train.sh`**  
   Zmień wartość parametru `--precond` na nazwę Twojego modelu, aby trening korzystał z nowo dodanej architektury i funkcji straty.

5. **Stwórz własny generator**
   Utwórz plik, który wczyta wagi modelu i będzie odpowiednio generował obrazki


W razie wątpliwości można skorzystać z przykładowego pull requesta (ddp-discrete) dostępnego w repozytorium.
