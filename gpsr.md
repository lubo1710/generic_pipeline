# Explanation

## Allgemeine Funktionsweise

Im Prinzip werden für jedes Feld, das nicht leer ist Informationen gesammelt und später ausgewertet,
sodass nur Objekte/Personen zurückgegeben werden, die die Einträge in der Query matchen. Durch Typkonflikte
müssen allerdings Regeln für die Query festgelegt werden um diesen aus dem Weg zu gehen.

Das ist der ObjectDesignator, auf den sich in den folgenden Kapiteln bezogen wird.

```
string uid
            string type 
            string[] shape
            ShapeSize[] shape_size
            string[] color
            string location
            string size
            geometry_msgs/PoseStamped[]
            string[] pose_source

            string[] attribute 

            string[] description
```

## Type
In type kann zwischen mehreren Wegen unterschieden werden.

### Leer
Falls das type Feld leer übergeben wird, werden alle erkannten **Objekte** zurückgegeben, die die 
anderen Spezifikationen erfüllen.

### Name von Objekt oder Typ
Falls das Feld mit einem Namen gefüllt ist, wird unterschieden ob dieser Name ein bekanntes Objekt ist
oder nicht. Falls es ein bekanntes Objekt ist wird nach diesem Objekt in der Szene gesucht und falls nicht
nach Personen. Sollte eine Person schon bekannt sein, kann diese unter der ID wiedererkannt werden und 
falls nicht wird sie abgespeichert unter einer ID und zurückgegeben.

Die List der erkannten Objekte muss in der Network klasse angepasst werden. Und in den Capabilities von dem 
*YoloAnnotator*

### Besonderheit bei person
Sollte 'person' im type feld stehen, werden alle Personen erkannt und zurückgegeben, falls alle anderen
spezifikation erfüllt sind. Dabei wird die nicht auf das Gesicht geachtet, sondern nur die Person erkannt.
Die Ausgabe enthält dabei einen leeren Type.

Der Ordner mit erkannten Gesichtern wird nicht nach jedem Durchlauf gelöscht. In dem
top-level directory liegt ein Skript, dass diesen Ordner löscht. Es kann hilfreich sein
den Ordner zu löschen, wenn viele Gesichter abgespeichert sind, um die Anzahl der Personen
mit denen das erkannte Gesicht verglichen wird zu reduzieren.

### Besonderheit für Gesichter
Sollte 'faces' im type Feld übergeben werden, werden alle Gesichter erkannt. Dabei wird eine eindeutige ID
festgelegt und unbekannte Gesichter werden unter dieser ID abgespeichert z.b. human_0 oder human_1, sodass in einem nächsten Aufruf,
das Gesicht wieder erkannt werden kann.

## Shape 
Wenn etwas in size drinsteht, wird eine ShapeSize zurückgegeben. Der Inhalt bei size
ist nicht wichtig. Die Rückgabe findet sich im ObjectDesignator unter ShapeSize. Die Rückgabe hat den Typ ShapeSize
und kommt als Liste.

## Color
Relativ Straightforward. Wenn color angegeben, muss die erkannte Farbe von dem Objekt der Farbe in der 
Query entsprechen. Falls nicht wird das Objekt nicht zurückgegeben.

## attribute
Bei attribute geht es um Spezifizierungen für die Klamotten und das Behaviour. Das Feld attributes ist eine Liste
wenn das behaviour erkannt werden soll, muss das gewünschte Verhalten, wie zum Beispiel 'sitting' an Stelle 0
der Liste gesetzt werden. Achtung wenn die Liste mehr als einen Eintrag hat, wird davon ausgegangen, dass 
Klamotten erkannt werden sollen.

Wenn klamotten erkannt werden sollen, gibt es immer eine Kombination aus Farbe und KLeidungsstück, dafür
soll die Farbe an 0. Stelle in der Liste stehen und das Klamottenstück an Stelle 1

## Nicht unterstützte Sachen
Alle nicht erwähnten Felder, werden nicht unterstützt und es kann keine Information darüber ermittelt werden

# Beispiele
Die Pose ist in jeder Ausgabe enthalten, wird hier zur Übersichtlichkeit allerdings ausgelassen
Color und alle Objekte:
```
type = ''
color = 'red'
==== result
[
ObjectDesignator 
    type = 'Metalmug'
    color = 'red' ,
ObjectDesignator 
    type = 'Crackerbox'
    color = 'red' ]
```

Alle stehenden Personen:
```
type = 'person'
attribute = ['standing']
==== result
[
ObjectDesignator 
    type = ''
    color = 'standing']
```

Objekte vom Typ Metalmug:
```
type = 'Metalmug'
==== result
[
ObjectDesignator 
    type = 'Metalmug']
```

Objekt mit Size:
```
type = ''
size = 'IRGENDWAS'
==== result
[
ObjectDesignator 
    type = 'Metalmug'
    shape_size = 
        dimensions: 
          x: 0.3567914733886719
          y: 0.10483702850341797
          z: 0.09453574314344526
        radius: 0
]
```

Bekannter Mensch:
```
type = 'faces'
==== result
[
ObjectDesignator 
    type = 'human_0'
]
```


