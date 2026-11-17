import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
import unicodedata
from tkinter import filedialog, messagebox, ttk
from urllib.parse import unquote, urlparse

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame
from ai import OllamaError, pull_ollama_model, suggest_semantic_tags_batch
from learning import LearningStoreError, get_relevant_feedback, record_tag_feedback
from mutagen import File as MutagenFile
from mutagen import MutagenError
from mutagen.id3 import COMM, ID3
from tkinterdnd2 import DND_FILES, TkinterDnD


AUDIO_EXTENSIONS = {".wav", ".mp3"}
AI_BATCH_SIZE = 3
IGNORED_FILENAME_TAGS = {"audio", "file", "sample", "sound", "wav", "mp3"}


def suggest_filename_tags(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", stem)
    stem = unicodedata.normalize("NFKC", stem)
    words = re.split(r"[\W_]+", stem)
    tags = []
    seen = set()
    for word in words:
        tag = word.strip().lower()
        if (
            not tag
            or (tag.isdigit() and (len(tag) < 3 or tag.startswith("0")))
            or tag in IGNORED_FILENAME_TAGS
            or tag in seen
        ):
            continue
        tags.append(tag)
        seen.add(tag)
    return tags


def combine_tags(original_tags, new_tags):
    combined = []
    seen = set()
    for tag in re.split(r"[,;\n]+", original_tags + "," + new_tags):
        tag = re.sub(r"\s+", " ", tag.replace("_", " ")).strip()
        key = tag.casefold()
        if tag and key not in seen:
            combined.append(tag)
            seen.add(key)
    return ", ".join(combined)


def read_comment(path):
    audio = MutagenFile(path)
    if audio is None:
        raise ValueError("Das Audioformat konnte nicht gelesen werden.")
    if audio.tags is None:
        return ""

    comments = audio.tags.getall("COMM")
    default_comment = next(
        (frame for frame in comments if frame.lang == "eng" and not frame.desc),
        None,
    )
    if default_comment is not None:
        return " ".join(str(value) for value in default_comment.text)
    if comments:
        return " ".join(str(value) for value in comments[0].text)
    return ""


def write_comment(path, comment):
    audio = MutagenFile(path)
    if audio is None:
        raise ValueError("Das Audioformat konnte nicht gelesen werden.")
    if audio.tags is None:
        audio.add_tags()

    tags = audio.tags
    if not isinstance(tags, ID3):
        raise ValueError("Die Datei unterstützt keine ID3-Kommentarmetadaten.")

    for frame in tags.getall("COMM"):
        if frame.lang == "eng" and not frame.desc:
            tags.delall(frame.HashKey)
    tags.add(COMM(encoding=3, lang="eng", desc="", text=comment))
    audio.save()


class TaggingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sound Library Tagger")
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        width = min(1120, screen_width - 80)
        height = min(650, screen_height - 120)
        x = max(0, (screen_width - width) // 2)
        y = max(0, (screen_height - height) // 2)
        self.root.geometry("{}x{}+{}+{}".format(width, height, x, y))
        self.root.minsize(min(width, 800), min(height, 500))
        self.files = {}
        self.ai_queue = queue.Queue()
        self.ai_running = False
        self.model_update_queue = queue.Queue()
        self.model_update_running = False
        self.ai_errors = []
        self.ai_success_count = 0
        self.player_channel = None
        self.player_path = None
        self.player_sound = None
        self.player_segment_sound = None
        self.player_raw = None
        self.player_duration = 0.0
        self.player_position = 0.0
        self.player_started_at = None
        self.player_seeking = False
        self.player_scale_updating = False
        self.player_after_id = None

        self._build_menu()
        self._build_ui()
        if sys.platform == "darwin":
            self.root.tk.createcommand(
                "::tk::mac::OpenDocument", self._handle_open_document
            )
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _handle_open_document(self, *paths):
        self.root.after_idle(self._open_paths, paths)

    def _open_paths(self, paths):
        normalized_paths = []
        for path in paths:
            if path.startswith("file://"):
                path = unquote(urlparse(path).path)
            normalized_paths.append(os.path.abspath(os.path.expanduser(path)))

        self._add_paths(normalized_paths)
        selected = [path for path in normalized_paths if path in self.files]
        if selected:
            self.tree.selection_set(selected)
            self.tree.focus(selected[0])
            self.tree.see(selected[0])
            self._generate_ai_tags()

    def _build_menu(self):
        menu_bar = tk.Menu(self.root)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(
            label="Dateien hinzufügen…",
            accelerator="⌘D",
            command=self._choose_files,
        )
        file_menu.add_command(
            label="Ordner hinzufügen…",
            accelerator="⌘O",
            command=self._choose_folder,
        )
        menu_bar.add_cascade(label="File", menu=file_menu)
        self.model_menu = tk.Menu(menu_bar, tearoff=0)
        self.model_menu.add_command(
            label="Auf Updates prüfen…",
            command=self._update_model,
        )
        menu_bar.add_cascade(label="Sprachmodell", menu=self.model_menu)
        self.root.configure(menu=menu_bar)
        self.root.bind("<Command-d>", self._on_add_files_shortcut)
        self.root.bind("<Command-o>", self._on_add_folder_shortcut)

    def _on_add_files_shortcut(self, _event):
        self._choose_files()
        return "break"

    def _on_add_folder_shortcut(self, _event):
        self._choose_folder()
        return "break"

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="Sound Library Tagger",
            font=("TkDefaultFont", 18, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text="Ziehe WAV-/MP3-Dateien oder einen Ordner hierher. "
            "Lokale KI leitet Tags aus Dateinamen und vorhandenen Tags ab. "
            "Originale bleiben bis zur Bestätigung unverändert.",
            wraplength=800,
        ).pack(anchor="w", pady=(4, 12))

        self.drop_area = ttk.Label(
            outer,
            text="Ordner oder Audiodateien hier ablegen",
            anchor="center",
            relief="groove",
            padding=18,
        )
        self.drop_area.pack(fill="x", pady=(0, 10))
        self.drop_area.drop_target_register(DND_FILES)
        self.drop_area.dnd_bind("<<Drop>>", self._on_drop)

        toolbar = ttk.Frame(outer)
        toolbar.pack(fill="x", pady=(0, 8))
        ttk.Button(
            toolbar, text="Ordner auswählen…", command=self._choose_folder
        ).pack(side="left")
        ttk.Button(
            toolbar, text="Dateien auswählen…", command=self._choose_files
        ).pack(side="left", padx=(8, 0))
        ttk.Label(toolbar, text="Ollama-Modell:").pack(side="left", padx=(16, 4))
        self.model_input = ttk.Entry(toolbar, width=16)
        self.model_input.insert(0, "qwen3:4b")
        self.model_input.pack(side="left")
        self.ai_button = ttk.Button(
            toolbar, text="KI-Tags erzeugen", command=self._generate_ai_tags
        )
        self.ai_button.pack(side="right")

        player_frame = ttk.Frame(outer)
        player_frame.pack(fill="x", pady=(0, 8))
        ttk.Button(
            player_frame, text="▶ Abspielen", command=self._play_selected_file
        ).pack(side="left")
        ttk.Button(
            player_frame, text="■ Stop", command=self._stop_playback
        ).pack(side="left", padx=(6, 12))
        self.player_filename = ttk.Label(
            player_frame, text="Keine Datei ausgewählt", width=28, anchor="w"
        )
        self.player_filename.pack(side="left", padx=(0, 8))
        self.player_scale = ttk.Scale(
            player_frame,
            from_=0,
            to=1,
            command=self._on_seek_change,
        )
        self.player_scale.pack(side="left", fill="x", expand=True)
        self.player_scale.bind("<ButtonPress-1>", self._begin_seek)
        self.player_scale.bind("<ButtonRelease-1>", self._end_seek)
        self.player_time_label = ttk.Label(
            player_frame, text="0:00 / 0:00", width=11, anchor="e"
        )
        self.player_time_label.pack(side="left", padx=(8, 0))

        columns = ("filename", "original_tags", "new_tags", "status")
        table_frame = ttk.Frame(outer)
        table_frame.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", selectmode="extended"
        )
        self.tree.heading("filename", text="Datei")
        self.tree.heading("original_tags", text="Ursprüngliche Tags")
        self.tree.heading("new_tags", text="Neue KI-Tags")
        self.tree.heading("status", text="Status")
        self.tree.column("filename", width=300, anchor="w")
        self.tree.column("original_tags", width=330, anchor="w")
        self.tree.column("new_tags", width=300, anchor="w")
        self.tree.column("status", width=140, anchor="w")
        vertical_scrollbar = ttk.Scrollbar(
            table_frame, orient="vertical", command=self.tree.yview
        )
        horizontal_scrollbar = ttk.Scrollbar(
            table_frame, orient="horizontal", command=self.tree.xview
        )
        self.tree.configure(
            yscrollcommand=vertical_scrollbar.set,
            xscrollcommand=horizontal_scrollbar.set,
        )
        self.tree.grid(row=0, column=0, sticky="nsew")
        vertical_scrollbar.grid(row=0, column=1, sticky="ns")
        horizontal_scrollbar.grid(row=1, column=0, sticky="ew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        self.tree.bind("<<TreeviewSelect>>", self._on_selection)
        self.file_context_menu = tk.Menu(self.root, tearoff=0)
        self.file_context_menu.add_command(
            label="Im Finder anzeigen", command=self._show_selected_in_finder
        )
        self.file_context_menu.add_command(
            label="Abspielen", command=self._play_selected_file
        )
        self.file_context_menu.add_separator()
        self.file_context_menu.add_command(
            label="Pfad kopieren", command=self._copy_selected_paths
        )
        self.file_context_menu.add_command(
            label="Dateiname kopieren", command=self._copy_selected_filenames
        )
        self.file_context_menu.add_separator()
        self.file_context_menu.add_command(
            label="Aus Liste entfernen", command=self._remove_selected_files
        )
        for sequence in ("<Button-2>", "<Button-3>", "<Control-Button-1>"):
            self.tree.bind(sequence, self._show_file_context_menu, add="+")

        edit_frame = ttk.Frame(outer)
        edit_frame.pack(fill="x", pady=(12, 4))
        ttk.Label(edit_frame, text="Neue Tags bearbeiten (durch Kommas getrennt):").pack(
            anchor="w"
        )
        self.tags_input = tk.Text(edit_frame, height=3, wrap="word")
        self.tags_input.pack(fill="x", pady=(4, 8))

        actions = ttk.Frame(outer)
        actions.pack(fill="x")
        ttk.Button(
            actions,
            text="Tags auf Auswahl anwenden",
            command=self._apply_draft,
        ).pack(side="left")
        ttk.Button(
            actions,
            text="Tags mit KI ergänzen",
            command=self._generate_related_tags,
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            actions,
            text="Tags in Dateien speichern…",
            command=self._save_comments,
        ).pack(side="right")

        self.status_label = ttk.Label(outer, text="Noch keine Dateien geladen.")
        self.status_label.pack(anchor="w", pady=(8, 0))

        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind("<<Drop>>", self._on_drop)

    def _show_file_context_menu(self, event):
        item = self.tree.identify_row(event.y)
        if not item:
            return "break"
        if item not in self.tree.selection():
            self.tree.selection_set(item)
        self.tree.focus(item)
        self.file_context_menu.entryconfigure(
            "Aus Liste entfernen",
            state="disabled" if self.ai_running else "normal",
        )
        try:
            self.file_context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.file_context_menu.grab_release()
        return "break"

    def _selected_file_paths(self):
        return list(self.tree.selection())

    def _context_file_path(self):
        focused = self.tree.focus()
        if focused in self.tree.selection():
            return focused
        selected = self._selected_file_paths()
        return selected[0] if selected else None

    def _show_selected_in_finder(self):
        paths = self._selected_file_paths()
        errors = []
        for path in paths:
            try:
                if sys.platform == "darwin":
                    subprocess.run(["open", "-R", path], check=True)
                elif sys.platform == "win32":
                    subprocess.run(["explorer", "/select,", path], check=True)
                else:
                    subprocess.run(
                        ["xdg-open", os.path.dirname(path)], check=True
                    )
            except (OSError, subprocess.CalledProcessError) as error:
                errors.append("{}: {}".format(os.path.basename(path), error))
        if errors:
            messagebox.showerror(
                "Datei-Manager konnte nicht geöffnet werden",
                "\n".join(errors[:10]),
                parent=self.root,
            )

    def _play_selected_file(self):
        path = self._context_file_path()
        if not path:
            messagebox.showinfo(
                "Keine Auswahl",
                "Wähle zuerst eine Audiodatei aus.",
                parent=self.root,
            )
            return

        if (
            self.player_path == path
            and self.player_channel is not None
            and self.player_channel.get_busy()
        ):
            return

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            if self.player_path != path:
                self._stop_playback(reset_position=False)
                sound = pygame.mixer.Sound(path)
                raw = sound.get_raw()
                frequency, sample_format, channels = pygame.mixer.get_init()
                frame_size = channels * (abs(sample_format) // 8)
                if frame_size <= 0 or frequency <= 0:
                    raise pygame.error("Ungültiges Audioformat für die Wiedergabe.")
                self.player_path = path
                self.player_sound = sound
                self.player_segment_sound = sound
                self.player_raw = raw
                self.player_duration = len(raw) / (frame_size * frequency)
                self.player_position = 0.0
                self.player_scale.configure(to=max(self.player_duration, 0.01))
                self.player_filename.configure(text=os.path.basename(path))
            if self.player_duration <= 0:
                raise pygame.error("Die Datei enthält keine abspielbaren Audiodaten.")
            if self.player_position >= self.player_duration:
                self.player_position = 0.0
            self._start_playback(self.player_position)
        except (OSError, pygame.error) as error:
            messagebox.showerror(
                "Wiedergabe nicht möglich",
                "{} konnte nicht abgespielt werden:\n{}".format(
                    os.path.basename(path), error
                ),
                parent=self.root,
            )

    def _start_playback(self, position):
        if position >= self.player_duration:
            self._stop_playback(reset_position=False)
            self.player_position = self.player_duration
            self._update_player_time(self.player_position)
            return
        if self.player_channel is None:
            self.player_channel = pygame.mixer.Channel(0)
        frequency, sample_format, channels = pygame.mixer.get_init()
        frame_size = channels * (abs(sample_format) // 8)
        frame_offset = int(position * frequency)
        byte_offset = frame_offset * frame_size
        if byte_offset:
            self.player_segment_sound = pygame.mixer.Sound(
                buffer=self.player_raw[byte_offset:]
            )
        else:
            self.player_segment_sound = self.player_sound
        self.player_channel.play(self.player_segment_sound)
        self.player_position = position
        self.player_started_at = time.monotonic()
        self._update_player_time(position)
        self._schedule_player_update()

    def _stop_playback(self, reset_position=True):
        if self.player_channel is not None:
            self.player_channel.stop()
        self.player_started_at = None
        if reset_position:
            self.player_position = 0.0
        self.player_segment_sound = self.player_sound
        self._update_player_time(self.player_position)

    def _begin_seek(self, _event):
        self.player_seeking = True

    def _on_seek_change(self, value):
        if self.player_path and not self.player_scale_updating:
            self._show_player_time(float(value))

    def _end_seek(self, _event):
        self.player_seeking = False
        if not self.player_path:
            return
        position = float(self.player_scale.get())
        if (
            self.player_channel is not None
            and self.player_channel.get_busy()
        ):
            self._start_playback(position)
        else:
            self.player_position = position
            self._update_player_time(position)

    def _update_player_time(self, position):
        position = max(0.0, min(position, self.player_duration))
        if not self.player_seeking:
            self.player_scale_updating = True
            try:
                self.player_scale.set(position)
            finally:
                self.player_scale_updating = False
        self._show_player_time(position)

    def _show_player_time(self, position):
        position = max(0.0, min(position, self.player_duration))
        self.player_time_label.configure(
            text="{} / {}".format(
                self._format_player_time(position),
                self._format_player_time(self.player_duration),
            )
        )

    @staticmethod
    def _format_player_time(seconds):
        total_seconds = max(0, int(seconds))
        minutes, seconds = divmod(total_seconds, 60)
        return "{}:{:02d}".format(minutes, seconds)

    def _schedule_player_update(self):
        if self.player_after_id is None:
            self.player_after_id = self.root.after(100, self._update_player)

    def _update_player(self):
        self.player_after_id = None
        if self.player_seeking:
            self._schedule_player_update()
            return
        if self.player_channel is not None and self.player_channel.get_busy():
            position = self.player_position + (
                time.monotonic() - self.player_started_at
            )
            self._update_player_time(position)
            self._schedule_player_update()
        elif self.player_started_at is not None:
            self.player_started_at = None
            self.player_position = self.player_duration
            self._update_player_time(self.player_duration)

    def _on_close(self):
        if self.player_after_id is not None:
            self.root.after_cancel(self.player_after_id)
            self.player_after_id = None
        self._stop_playback()
        if pygame.mixer.get_init():
            pygame.mixer.quit()
        self.root.destroy()

    def _update_model(self):
        if self.model_update_running:
            return
        if self.ai_running:
            messagebox.showwarning(
                "KI läuft",
                "Warte, bis die aktuelle KI-Verarbeitung abgeschlossen ist.",
                parent=self.root,
            )
            return

        model = self.model_input.get().strip()
        if not model:
            messagebox.showwarning(
                "Modell fehlt",
                "Gib einen installierten Ollama-Modellnamen ein.",
                parent=self.root,
            )
            return
        if not messagebox.askyesno(
            "Sprachmodell aktualisieren",
            "Soll Ollama das Modell '{}' auf Updates prüfen und bei Bedarf "
            "Modelldaten herunterladen? Das kann mehrere GB umfassen. "
            "Die App muss dafür nicht beendet werden.".format(model),
            parent=self.root,
        ):
            return

        self.model_update_running = True
        self.model_menu.entryconfigure(0, state="disabled")
        self.model_input.configure(state="disabled")
        self.ai_button.configure(state="disabled")
        self.status_label.configure(
            text="Prüfe Updates für das Sprachmodell {}…".format(model)
        )
        threading.Thread(
            target=self._run_model_update, args=(model,), daemon=True
        ).start()
        self.root.after(100, self._poll_model_update_results)

    def _run_model_update(self, model):
        try:
            pull_ollama_model(
                model,
                lambda status: self.model_update_queue.put(("progress", status)),
            )
        except OllamaError as error:
            self.model_update_queue.put(("error", str(error)))
        else:
            self.model_update_queue.put(("success",))

    def _poll_model_update_results(self):
        result = None
        while True:
            try:
                update = self.model_update_queue.get_nowait()
            except queue.Empty:
                break

            if update[0] == "progress":
                self.status_label.configure(
                    text="Sprachmodell: {}".format(update[1])
                )
            else:
                result = update

        if result is None:
            self.root.after(100, self._poll_model_update_results)
            return

        self.model_update_running = False
        self.model_menu.entryconfigure(0, state="normal")
        self.model_input.configure(state="normal")
        self.ai_button.configure(state="normal")
        if result[0] == "error":
            self.status_label.configure(text="Sprachmodell-Update fehlgeschlagen.")
            messagebox.showerror(
                "Sprachmodell-Update fehlgeschlagen",
                result[1],
                parent=self.root,
            )
            return

        self.status_label.configure(
            text="Sprachmodell {} ist aktuell.".format(self.model_input.get().strip())
        )
        messagebox.showinfo(
            "Sprachmodell aktuell",
            "Das Modell wurde erfolgreich geprüft bzw. aktualisiert. "
            "Ein Neustart der App ist nicht erforderlich.",
            parent=self.root,
        )

    def _copy_selected_paths(self):
        self._copy_to_clipboard(self._selected_file_paths())

    def _copy_selected_filenames(self):
        names = [
            os.path.basename(path) for path in self._selected_file_paths()
        ]
        self._copy_to_clipboard(names)

    def _copy_to_clipboard(self, values):
        if not values:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append("\n".join(values))
        self.status_label.configure(
            text="{} Eintrag/Einträge in die Zwischenablage kopiert.".format(
                len(values)
            )
        )

    def _remove_selected_files(self):
        if self.ai_running:
            return
        paths = self._selected_file_paths()
        if not paths:
            return
        if any(self.files[path]["status"] == "Zu schreiben" for path in paths):
            if not messagebox.askyesno(
                "Einträge entfernen",
                "Nicht gespeicherte Entwürfe gehen aus der Liste verloren. "
                "Die Audiodateien selbst bleiben unverändert. Fortfahren?",
                parent=self.root,
            ):
                return
        for path in paths:
            del self.files[path]
            self.tree.delete(path)
        if self.player_path in paths:
            self._stop_playback()
            self.player_path = None
            self.player_sound = None
            self.player_segment_sound = None
            self.player_raw = None
            self.player_duration = 0.0
            self.player_scale.configure(to=1)
            self.player_filename.configure(text="Keine Datei ausgewählt")
            self._update_player_time(0)
        self.status_label.configure(
            text="{} Eintrag/Einträge aus der Liste entfernt.".format(len(paths))
        )

    def _choose_folder(self):
        folder = filedialog.askdirectory(title="Sound-Library auswählen")
        if folder:
            self._add_paths([folder])

    def _choose_files(self):
        paths = filedialog.askopenfilenames(
            title="Audiodateien auswählen",
            filetypes=[("Audiodateien", "*.wav *.mp3"), ("Alle Dateien", "*")],
        )
        self._add_paths(paths)

    def _on_drop(self, event):
        self._add_paths(self.root.tk.splitlist(event.data))
        return event.action

    def _add_paths(self, paths):
        candidates = []
        for path in paths:
            path = os.path.abspath(path)
            if os.path.isdir(path):
                for directory, _, filenames in os.walk(path):
                    candidates.extend(
                        os.path.join(directory, filename) for filename in filenames
                    )
            else:
                candidates.append(path)

        added = 0
        errors = []
        for path in candidates:
            if os.path.splitext(path)[1].lower() not in AUDIO_EXTENSIONS:
                continue
            if path in self.files:
                continue

            try:
                current_comment = read_comment(path)
                status = "Kommentar vorhanden" if current_comment else "Zu prüfen"
            except (OSError, MutagenError, ValueError) as error:
                current_comment = ""
                status = "Metadatenfehler"
                errors.append("{}: {}".format(os.path.basename(path), error))

            original_tags = current_comment or ", ".join(suggest_filename_tags(path))
            self.files[path] = {
                "original_tags": original_tags,
                "tags": "",
                "status": status,
            }
            self.tree.insert(
                "",
                "end",
                iid=path,
                values=(
                    os.path.basename(path),
                    original_tags or "—",
                    "",
                    status,
                ),
            )
            added += 1

        self.status_label.configure(
            text="{} Datei(en) geladen. WAV- und MP3-Dateien werden unterstützt."
            .format(len(self.files))
        )
        if errors:
            messagebox.showwarning(
                "Einige Metadaten konnten nicht gelesen werden",
                "\n".join(errors[:10]),
                parent=self.root,
            )
        elif added == 0:
            self.status_label.configure(
                text="Keine neuen WAV- oder MP3-Dateien gefunden."
            )

    def _on_selection(self, _event):
        selected = self.tree.selection()
        self.tags_input.delete("1.0", "end")
        if selected:
            self.tags_input.insert("1.0", self.files[selected[0]]["tags"])

    def _generate_related_tags(self):
        seed_tags = self.tags_input.get("1.0", "end").strip()
        if not seed_tags:
            messagebox.showwarning(
                "Keine Vorlage",
                "Gib zuerst eigene Tags ein, die als Vorlage dienen sollen.",
                parent=self.root,
            )
            return
        self._generate_ai_tags(seed_tags=seed_tags)

    def _generate_ai_tags(self, seed_tags=None):
        if self.ai_running:
            return
        if self.model_update_running:
            messagebox.showwarning(
                "Sprachmodell wird aktualisiert",
                "Warte, bis das Sprachmodell-Update abgeschlossen ist.",
                parent=self.root,
            )
            return

        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo(
                "Keine Auswahl",
                "Wähle zuerst die Dateien aus, für die Tags erzeugt werden sollen.",
                parent=self.root,
            )
            return

        model = self.model_input.get().strip()
        if not model:
            messagebox.showwarning(
                "Modell fehlt",
                "Gib einen installierten Ollama-Modellnamen ein.",
                parent=self.root,
            )
            return

        self.ai_running = True
        self.ai_errors = []
        self.ai_success_count = 0
        self.ai_no_tags_count = 0
        self.ai_button.configure(state="disabled")
        for path in selected:
            self.files[path]["status"] = "KI läuft…"
            self.files[path]["learnable"] = False
            self.tree.set(path, "status", "KI läuft…")
        jobs = [
            (path, os.path.basename(path), self.files[path]["original_tags"])
            for path in selected
        ]
        self.status_label.configure(
            text=(
                "Ergänze deine Tags mit lokalen KI-Vorschlägen für "
                "{} Datei(en)…".format(len(selected))
                if seed_tags
                else "Erzeuge lokale KI-Tags für {} Datei(en) in Gruppen…".format(
                    len(selected)
                )
            )
        )
        threading.Thread(
            target=self._run_ai, args=(jobs, model, seed_tags), daemon=True
        ).start()
        self.root.after(100, self._poll_ai_results)

    def _run_ai(self, jobs, model, seed_tags=None):
        for start in range(0, len(jobs), AI_BATCH_SIZE):
            batch = jobs[start : start + AI_BATCH_SIZE]
            batch_files = [
                (index, filename, original_tags)
                for index, (_, filename, original_tags) in enumerate(batch)
            ]
            try:
                learning_context = {
                    index: (
                        combine_tags(original_tags, seed_tags)
                        if seed_tags
                        else original_tags
                    )
                    for index, (_, _, original_tags) in enumerate(batch)
                }
                learning_preferences = {
                    index: get_relevant_feedback(
                        filename, learning_context[index]
                    )
                    for index, (_, filename, _) in enumerate(batch)
                }
                options = {"learning_preferences": learning_preferences}
                if seed_tags:
                    options["seed_tags"] = {
                        index: seed_tags for index in range(len(batch))
                    }
                tags_by_id = suggest_semantic_tags_batch(
                    batch_files,
                    model,
                    **options
                )
            except (OllamaError, LearningStoreError) as error:
                for path, _, _ in batch:
                    self.ai_queue.put(("error", path, str(error)))
            else:
                for index, (path, _, _) in enumerate(batch):
                    tags = tags_by_id[index]
                    if seed_tags:
                        self.ai_queue.put(
                            (
                                "success",
                                path,
                                combine_tags(seed_tags, ", ".join(tags)).split(
                                    ", "
                                ),
                                bool(tags),
                            )
                        )
                    else:
                        self.ai_queue.put(("success", path, tags))
            self.ai_queue.put(("progress", min(start + len(batch), len(jobs)), len(jobs)))
        self.ai_queue.put(("done",))

    def _poll_ai_results(self):
        done = False
        while True:
            try:
                result = self.ai_queue.get_nowait()
            except queue.Empty:
                break

            if result[0] == "success":
                _, path, tags = result[:3]
                has_suggestions = result[3] if len(result) > 3 else bool(tags)
                self.files[path]["tags"] = ", ".join(tags)
                self.tree.set(path, "new_tags", self.files[path]["tags"])
                if has_suggestions:
                    self.files[path]["status"] = "KI-Vorschlag"
                    self.files[path]["learnable"] = True
                    self.tree.set(path, "status", "KI-Vorschlag")
                    self.ai_success_count += 1
                else:
                    self.files[path]["status"] = "Keine neuen Tags"
                    self.files[path]["learnable"] = False
                    self.tree.set(path, "status", "Keine neuen Tags")
                    self.ai_no_tags_count += 1
            elif result[0] == "error":
                _, path, error = result
                self.files[path]["status"] = "KI-Fehler"
                self.tree.set(path, "status", "KI-Fehler")
                self.ai_errors.append(
                    "{}: {}".format(os.path.basename(path), error)
                )
            elif result[0] == "progress":
                _, completed, total = result
                self.status_label.configure(
                    text="KI-Tags: {} von {} Datei(en) verarbeitet…".format(
                        completed, total
                    )
                )
            else:
                done = True

        if not done:
            self.root.after(100, self._poll_ai_results)
            return

        self.ai_running = False
        self.ai_button.configure(state="normal")
        if self.ai_errors:
            self.status_label.configure(
                text="KI-Tags: {} Vorschläge, {} ohne neue Tags, {} Fehler."
                .format(
                    self.ai_success_count,
                    self.ai_no_tags_count,
                    len(self.ai_errors),
                )
            )
            messagebox.showerror(
                "KI-Tags konnten nicht vollständig erzeugt werden",
                "\n".join(self.ai_errors[:10]),
                parent=self.root,
            )
        elif self.ai_no_tags_count:
            if self.ai_success_count:
                status = (
                    "KI-Tags: {} Vorschläge; für {} Datei(en) gab es keine "
                    "neuen Tags."
                ).format(self.ai_success_count, self.ai_no_tags_count)
            else:
                status = "Für {} Datei(en) wurden keine neuen Tags vorgeschlagen.".format(
                    self.ai_no_tags_count
                )
            self.status_label.configure(text=status)
        elif self.ai_success_count:
            self.status_label.configure(
                text="KI-Tags erzeugt. Bitte Vorschläge prüfen und anwenden."
            )
        elif self.ai_success_count == 0:
            self.status_label.configure(text="Keine KI-Tags erzeugt.")

    def _apply_draft(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo(
                "Keine Auswahl", "Wähle zuerst mindestens eine Datei aus.", parent=self.root
            )
            return

        draft = self.tags_input.get("1.0", "end").strip()
        feedback = [
            (
                os.path.basename(path),
                self.files[path]["original_tags"],
                self.files[path]["tags"],
                draft,
            )
            for path in selected
            if self.files[path].get("learnable")
        ]
        learning_error = None
        learned_feedback = 0
        if feedback:
            try:
                learned_feedback = record_tag_feedback(feedback)
            except LearningStoreError as error:
                learning_error = str(error)

        for path in selected:
            self.files[path]["tags"] = draft
            self.files[path]["status"] = "Zu schreiben"
            self.tree.set(path, "new_tags", draft)
            self.tree.set(path, "status", "Zu schreiben")
        status = (
            "Tags auf {} Datei(en) angewendet. Noch nicht gespeichert: "
            "Klicke auf „Tags in Dateien speichern…“.".format(len(selected))
        )
        if learned_feedback:
            status += " Korrekturen lokal für ähnliche Vorschläge gemerkt."
        self.status_label.configure(text=status)
        if learning_error:
            messagebox.showwarning(
                "Lernkorrekturen nicht gespeichert",
                "Die Tags wurden angewendet, aber deine Korrekturen konnten "
                "nicht für künftige Vorschläge gespeichert werden:\n{}".format(
                    learning_error
                ),
                parent=self.root,
            )

    def _save_comments(self):
        pending = [
            path
            for path, data in self.files.items()
            if data["status"] == "Zu schreiben"
        ]
        if not pending:
            messagebox.showinfo(
                "Nichts zu schreiben",
                "Wähle Dateien aus, bearbeite die Tags und klicke zuerst auf "
                "„Tags auf Auswahl anwenden“.",
                parent=self.root,
            )
            return

        empty = [path for path in pending if not self.files[path]["tags"].strip()]
        if empty:
            messagebox.showwarning(
                "Leere Tags",
                "Bitte gib für jede Datei Tags ein. Dateien ohne Tags wurden nicht geändert.",
                parent=self.root,
            )
            return

        if not messagebox.askyesno(
            "Metadaten ändern",
            "Die Tags werden als Kommentare in {} Audiodatei(en) gespeichert. "
            "Fortfahren?".format(len(pending)),
            parent=self.root,
        ):
            return

        errors = []
        saved = 0
        for path in pending:
            try:
                comment = combine_tags(
                    self.files[path]["original_tags"], self.files[path]["tags"]
                )
                write_comment(path, comment)
            except (OSError, MutagenError, ValueError) as error:
                self.files[path]["status"] = "Fehler"
                self.tree.set(path, "status", "Fehler")
                errors.append("{}: {}".format(os.path.basename(path), error))
            else:
                self.files[path]["status"] = "Gespeichert"
                self.tree.set(path, "status", "Gespeichert")
                saved += 1

        self.status_label.configure(
            text="Tags in {} Datei(en) gespeichert.".format(saved)
        )
        if errors:
            messagebox.showerror(
                "Nicht alle Tags wurden gespeichert",
                "{} erfolgreich.\n\n{}".format(saved, "\n".join(errors[:10])),
                parent=self.root,
            )
        else:
            messagebox.showinfo(
                "Fertig",
                "Tags in {} Datei(en) gespeichert. Falls die Dateien in Sononym "
                "bereits geladen sind, aktualisiere dort die Bibliothek."
                .format(saved),
                parent=self.root,
            )


def main():
    root = TkinterDnD.Tk()
    app = TaggingApp(root)
    if len(sys.argv) > 1:
        root.after_idle(app._open_paths, sys.argv[1:])
    root.mainloop()


if __name__ == "__main__":
    main()
