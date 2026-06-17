öffnen comando palette 
Cmd + Shift + P (oder als Langform Befehlstaste + Umschalttaste + P)


1. Erstelle eine virtuelle Umgebung:

python3 -m venv path/to/venv
Ersetze path/to/venv durch den gewünschten Speicherort für deine virtuelle Umgebung (z. B. ~/myenv).

2. Aktiviere die virtuelle Umgebung:
Aktiviere die virtuelle Umgebung mit dem folgenden Befehl:

source .venv/bin/activate


3. Installiere ipykernel in der virtuellen Umgebung:
Sobald die virtuelle Umgebung aktiviert ist, kannst du den Befehl ausführen, um ipykernel zu installieren:

bash
Code kopieren
pip install ipykernel
4. Füge die virtuelle Umgebung zu Jupyter hinzu:
Wenn du ipykernel installiert hast, kannst du die virtuelle Umgebung als Kernel zu Jupyter hinzufügen:

bash
Code kopieren
python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"
Ersetze myenv durch den Namen deiner virtuellen Umgebung.

5. Starte Jupyter Notebook:
Jetzt kannst du dein Jupyter Notebook starten, und es sollte dir die neue Python-Umgebung als Option im Kernel-Menü anzeigen:

bash
Code kopieren
jupyter notebook
Mit dieser Vorgehensweise solltest du ipykernel erfolgreich installieren und verwenden können, ohne auf die systemweite Paketinstallation angewiesen zu sein.






