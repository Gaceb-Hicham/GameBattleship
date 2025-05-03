# GameBattleship
Agent training
### README: Battleship Game with Q-Learning Agent

This project implements a **Battleship game** environment with a graphical user interface (GUI) and a Q-learning agent to play the game autonomously. Below is a brief overview of each class and the main script, suitable for a README file.

---

### Classes Overview

1. **BattleshipEnv (BattleshipEnv.py)**  
   - Implements a custom Gym environment for the Battleship game.  
   - Creates a grid (default 10x10) with randomly placed ships of varying sizes (e.g., 2x1, 3x1, 5x1, 5x2).  
   - Tracks player shots, hits, misses, and ship statuses (sunk or not).  
   - Provides rewards based on actions (e.g., +1 for a hit, +20 for sinking a ship, -0.05 for a miss).  
   - Supports game termination when all ships are sunk or after a maximum number of steps (300).

2. **BattleshipGUI (BattleshipGUI.py)**  
   - Creates a graphical interface using Tkinter for the Battleship game.  
   - Displays a grid of buttons representing the game board, where players can click to fire shots.  
   - Updates the board visually (e.g., red for hits, blue for misses, green for sunk ships).  
   - Allows players to play manually, against a trained agent, or alternate turns with the agent.  
   - Includes options to reset the game and display game status (e.g., "Hit!", "You sunk a ship!").

3. **QLearningAgent (QLearningAgent.py)**  
   - Implements a Q-learning reinforcement learning agent to play the Battleship game.  
   - Learns an optimal strategy by exploring the environment and updating a Q-table based on rewards.  
   - Uses an epsilon-greedy policy to balance exploration and exploitation.  
   - Saves training metrics (e.g., total reward, hit rate, steps) to a CSV file and periodically saves the Q-table.  
   - Includes experience replay and Q-table pruning to improve learning efficiency.

4. **Training Metrics Visualization (training_metrics.py)**  
   - Visualizes training performance of the Q-learning agent using data from the CSV file.  
   - Generates plots for total reward, hit rate, and steps per episode.  
   - Saves the plots as PNG files for analysis (e.g., `total_reward.png`, `hit_rate.png`).

---

### Main Script (MainGame.py)

- **Purpose**: Orchestrates the Battleship game by initializing the environment, training the agent, and launching the GUI.  
- **Flow**:  
  1. Creates a `BattleshipEnv` for training with a 10x10 grid.  
  2. Initializes a `QLearningAgent` with learning parameters (e.g., learning rate = 0.1, discount factor = 0.95).  
  3. Trains the agent for 10 episodes (can be adjusted for more training).  
  4. Creates a new `BattleshipEnv` for the GUI and launches the `BattleshipGUI` with the trained agent.  
- **Outcome**: Allows users to interact with the game manually or observe the trained agent playing.

---

This project combines reinforcement learning with an interactive game, making it a great example of applying AI to classic games. For detailed usage, adjust training episodes in `MainGame.py` or explore the GUI options to play against the agent.



### ملف README: لعبة باتلشيب مع وكيل تعلم Q-Learning

هذا المشروع ينفذ لعبة **باتلشيب** بيئيًا مع واجهة مستخدم رسومية (GUI) ووكيل تعلم Q-Learning للعب بشكل تلقائي. فيما يلي نظرة عامة موجزة لكل فئة (Class) والملف الرئيسي، مناسبة لملف README باللغة العربية.

---

### نظرة عامة على الفئات

1. **BattleshipEnv (BattleshipEnv.py)**  
   - ينفذ بيئة مخصصة للعبة باتلشيب باستخدام مكتبة Gym.  
   - ينشئ شبكة (افتراضيًا 10x10) مع سفن موضوعة عشوائيًا بأحجام مختلفة (مثل 2x1، 3x1، 5x1، 5x2).  
   - يتتبع طلقات اللاعب، الإصابات، الإخفاقات، وحالة السفن (غارقة أو لا).  
   - يوفر مكافآت بناءً على الإجراءات (مثل +1 للإصابة، +20 لغرق سفينة، -0.05 للإخفاق).  
   - يدعم إنهاء اللعبة عند غرق جميع السفن أو بعد الحد الأقصى للخطوات (300).

2. **BattleshipGUI (BattleshipGUI.py)**  
   - ينشئ واجهة رسومية باستخدام Tkinter للعبة باتلشيب.  
   - يعرض شبكة من الأزرار تمثل لوحة اللعبة، حيث يمكن للاعبين النقر لإطلاق الطلقات.  
   - يحدّث اللوحة بصريًا (مثل الأحمر للإصابات، الأزرق للإخفاقات، الأخضر للسفن الغارقة).  
   - يتيح اللعب يدويًا، أو ضد وكيل مدرب، أو بالتناوب مع الوكيل.  
   - يتضمن خيارات لإعادة تعيين اللعبة وعرض حالة اللعبة (مثل "إصابة!"، "لقد أغرقت سفينة!").

3. **QLearningAgent (QLearningAgent.py)**  
   - ينفذ وكيل تعلم تقوية (Q-Learning) للعب لعبة باتلشيب.  
   - يتعلم استراتيجية مثالية من خلال استكشاف البيئة وتحديث جدول Q بناءً على المكافآت.  
   - يستخدم سياسة epsilon-greedy لتحقيق توازن بين الاستكشاف والاستغلال.  
   - يحفظ مقاييس التدريب (مثل المكافأة الكلية، معدل الإصابة، الخطوات) في ملف CSV ويحفظ جدول Q بشكل دوري.  
   - يتضمن إعادة تشغيل التجربة وتقليم جدول Q لتحسين كفاءة التعلم.

4. **Training Metrics Visualization (training_metrics.py)**  
   - يصور أداء تدريب وكيل Q-Learning باستخدام بيانات من ملف CSV.  
   - يولد رسومًا بيانية للمكافأة الكلية، معدل الإصابة، والخطوات لكل حلقة.  
   - يحفظ الرسوم كملفات PNG للتحليل (مثل `total_reward.png`، `hit_rate.png`).

---

### الملف الرئيسي (MainGame.py)

- **الغرض**: ينسق لعبة باتلشيب من خلال تهيئة البيئة، تدريب الوكيل، وتشغيل الواجهة الرسومية.  
- **التدفق**:  
  1. ينشئ `BattleshipEnv` للتدريب مع شبكة 10x10.  
  2. يهيئ `QLearningAgent` بمعاملات التعلم (مثل معدل التعلم = 0.1، عامل الخصم = 0.95).  
  3. يدرب الوكيل لمدة 10 حلقات (يمكن تعديلها لتدريب أكثر).  
  4. ينشئ `BattleshipEnv` جديدة للواجهة الرسومية ويشغل `BattleshipGUI` مع الوكيل المدرب.  
- **النتيجة**: يتيح للمستخدمين التفاعل مع اللعبة يدويًا أو مشاهدة الوكيل المدرب وهو يلعب.

---

هذا المشروع يجمع بين التعلم التقوية ولعبة تفاعلية، مما يجعله مثالًا رائعًا لتطبيق الذكاء الاصطناعي على الألعاب الكلاسيكية. للاستخدام التفصيلي، اضبط عدد حلقات التدريب في `MainGame.py` أو استكشف خيارات الواجهة الرسومية للعب ضد الوكيل.
