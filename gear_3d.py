from typing import Tuple, List, Dict, Set, Any
import math
import glfw
import glm
from OpenGL.GL import *
import ctypes
from gear import make_gear_triangles


class Model:
    """
    モデル用の基底クラス
    基本的な使い方としては、コンストラクタに頂点座標・頂点色・インデックスの
    3配列を渡して初期化し、draw()で描画する。
    また、rotate/translateで姿勢・位置を変化させ、その結果をget_model_matrixで取得し、
    シェーダに渡して反映させる
    """

    def __init__(self, vertices: List[float], colors: List[float], indices: List[int]) -> None:
        """
        モデルを生成する
        :param vertices: 頂点座標を格納したリスト。1頂点毎にx,y,zの3つの値を含む
        :param colors: 頂点色を格納したリスト。1頂点毎にr,g,bの3つの値を含む
        :param indices: インデックスを格納したリスト。三角形1つ毎に3つの頂点番号を含む
        """
        self.position = [0.0, 0.0, 0.0]  # 位置座標
        self.rotation = [0.0, 0.0, 0.0]  # 各軸毎の回転角
        self.num_indices = len(indices)

        # ctypesを使ってデータをOpenGLに渡せる形式(生のバイト列)に変換する
        VertexArray = ctypes.c_float * len(vertices)
        c_vertices = VertexArray(*vertices)  # 生のバイト列生成
        ColorArray = ctypes.c_float * len(colors)
        c_colors = ColorArray(*colors)
        IndexArray = ctypes.c_uint32 * len(indices)
        c_indices = IndexArray(*indices)

        # VBO(GPUに置かれるメモリバッファ)を生成する
        buffers = glGenBuffers(3)  # 頂点座標・頂点色・インデックス用のバッファを確保
        # 頂点情報
        glBindBuffer(GL_ARRAY_BUFFER, buffers[0])
        glBufferData(GL_ARRAY_BUFFER, ctypes.sizeof(c_vertices), c_vertices, GL_STATIC_DRAW)
        # 頂点色
        glBindBuffer(GL_ARRAY_BUFFER, buffers[1])
        glBufferData(GL_ARRAY_BUFFER, ctypes.sizeof(c_colors), c_colors, GL_STATIC_DRAW)
        # インデックス
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[2])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, ctypes.sizeof(c_indices), c_indices, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        # VAO(VBOをひとまとめで管理できるもの)を生成する
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)
        glEnableVertexAttribArray(0)  # 0番の属性を有効化(VertexシェーダのVertexPositionに対応)
        glEnableVertexAttribArray(1)  # 1番の属性を有効化(VertexシェーダのVertexColorに対応)
        glBindBuffer(GL_ARRAY_BUFFER, buffers[0])  # VAO-0番に頂点座標VBOを紐づけ
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, buffers[1])  # VAO-1番に頂点色VBOを紐づけ
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[2])  # VAOにインデックスVBOを紐づけ
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        self.vao = vao

    def draw(self) -> None:
        """
        モデルを描画する
        """
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.num_indices, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def reset_position(self, x: float, y: float, z: float) -> None:
        """
        モデルの位置を指定する
        :param x: X座標
        :param y: Y座標
        :param z: Z座標
        """
        self.position = [x, y, z]

    def reset_rotation(self, x: float, y: float, z: float) -> None:
        """
        モデルの姿勢を(オイラー角で)指定する
        :param x: X軸周りの回転角(degree)
        :param y: Y軸周りの回転角(degree)
        :param z: Z軸周りの回転角(degree)
        """
        self.rotation = [x, y, z]

    def translate(self, diff_x: float = 0.0, diff_y: float = 0.0, diff_z: float = 0.0) -> None:
        """
        モデルの位置を相対的に変化させる
        :param diff_x: X軸方向の移動量
        :param diff_y: Y軸方向の移動量
        :param diff_z: Z軸方向の移動量
        """
        self.position[0] += diff_x
        self.position[1] += diff_y
        self.position[2] += diff_z

    def rotate(self, diff_x: float = 0.0, diff_y: float = 0.0, diff_z: float = 0.0) -> None:
        """
        モデルの姿勢を相対的に変化させる
        :param diff_x: X軸周りの回転量(degree)
        :param diff_y: Y軸周りの回転量(degree)
        :param diff_z: Z軸周りの回転量(degree)
        """
        self.rotation[0] = math.fmod(self.rotation[0] + diff_x, 360.0)
        self.rotation[1] = math.fmod(self.rotation[1] + diff_y, 360.0)
        self.rotation[2] = math.fmod(self.rotation[2] + diff_z, 360.0)

    def get_model_matrix(self) -> "glm.mat4x4":
        """
        モデル行列を取得する
        :return: モデル行列を表す4x4正方行列
        """
        rx, ry, rz = self.rotation
        m = glm.mat4(1.0)
        m = glm.translate(m, glm.vec3(*self.position))
        m = glm.rotate(m, math.radians(rz), glm.vec3(0.0, 0.0, 1.0))
        m = glm.rotate(m, math.radians(ry), glm.vec3(0.0, 1.0, 0.0))
        m = glm.rotate(m, math.radians(rx), glm.vec3(1.0, 0.0, 0.0))
        return m


class SimpleModel(Model):
    """
    Modelの派生例(4角形を描くサンプル)
    """
    def __init__(self):
        # データ
        vertices = [  # 頂点情報
            10.0, 0.0, 10.0,
            10.0, 0.0, -10.0,
            -10.0, 0.0, -10.0,
            -10.0, 0.0, 10.0,
        ]
        colors = [  # 色情報
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]
        indices = [  # インデックス情報
            0, 1, 2, 2, 3, 0
        ]
        super().__init__(vertices, colors, indices)


class GearModel(Model):
    """
    歯車用のモデルクラス(gear_anim.pyのGearクラスのOpenGL向けアレンジ)
    X-Z平面上に歯車を配置し、Y軸が奥行となっている
    コンストラクタで生成したのち、
    駆動歯車ならset_center/set_angle/set_velocityで位置・角度・角速度を設定する
    従動歯車ならadjust_center/set_angleで位置・角度を設定する
    """

    def __init__(self, params: Dict[str, Any], driver: "GearModel" = None) -> None:
        """
        歯車を生成する
        :param params: 歯車のパラメータを表す辞書。詳細はgear.make_gear_triangles、
        およびpython gear.py --helpを参照のこと
        本関数独自パラメータとして以下のものがある。
         - r : 歯車の色(R要素: 0.0～1.0)
         - g : 歯車の色(G要素: 0.0～1.0)
         - b : 歯車の色(B要素: 0.0～1.0)
        :param driver: 本歯車に影響を及ぼす駆動歯車。自らが駆動歯車の場合はNone
        """
        vertices, indices = make_gear_triangles(**params)
        num_vertices = len(vertices) // 3
        col_r = params.get("r", 1.0)
        col_g = params.get("g", 1.0)
        col_b = params.get("b", 1.0)
        colors = [col_r, col_g, col_b] * num_vertices
        super().__init__(vertices, colors, indices)

        self.m = params['m']
        self.z = params['z']
        self.radius = self.m * self.z / 2
        self.angle = 0.0
        self.velocity = 0.0

        self.cx = 0
        self.cy = 0
        self.driver = driver
        self.driven_set: Set['GearModel'] = set()
        if driver:
            driver.add_driven(self)
            self.adjust_velocity()

    def add_driven(self, driven: 'GearModel') -> None:
        """
        従動歯車を登録する。この情報は自らの速度が変更された時に、
        従動歯車の速度も合わせて更新するために用いる
        :param driven: 従動歯車
        """
        self.driven_set.add(driven)

    def set_center(self, xy: Tuple[float, float]) -> None:
        """
        歯車中心の描画位置を指定する
        :param xy: 歯車の中心座標(x, y)。単位はpixel
        """
        self.cx = xy[0]
        self.cy = xy[1]

    def adjust_center(self, direction_angle: float) -> None:
        """
        駆動歯車に合わせて自らの位置を決定する
        :param direction_angle: 相手の歯車に対する自身の方位角(degree)
        """
        if self.driver is None:
            raise ValueError('Parent gear is not set.')
        a = self.radius + self.driver.radius  # 中心距離
        t = math.radians(direction_angle)
        self.cx = self.driver.cx + a * math.cos(t)
        self.cy = self.driver.cy + a * math.sin(t)

    def set_angle(self, angle: float) -> None:
        """
        歯車の回転角を指定する
        :param angle: 回転角(degree)
        """
        self.angle = angle

    def set_velocity(self, velocity: float) -> None:
        """
        歯車の角速度を指定する
        :param velocity: 角速度(degree per frame)
        :return:
        """
        self.velocity = velocity
        # 従動歯車も合わせて更新する
        for driven in self.driven_set:
            driven.adjust_velocity()

    def adjust_velocity(self) -> None:
        """
        駆動歯車に合わせて自らの角速度を決定する
        """
        if self.driver is None:
            raise ValueError('Parent gear is not set.')
        self.velocity = -self.driver.velocity * self.driver.z / self.z
        # 従動歯車も合わせて更新する
        for driven in self.driven_set:
            driven.adjust_velocity()

    def update(self) -> None:
        """
        歯車の角度を更新する
        """
        self.angle += self.velocity

    def get_model_matrix(self):
        """
        モデル行列を取得する
        :return: モデル行列を表す4x4正方行列
        """
        rx, ry, rz = self.rotation
        tx, ty, tz = self.position
        ry_mod = ty + self.angle
        position_mod = [tx + self.cx, ty, tz + self.cy]
        m = glm.mat4(1.0)
        m = glm.translate(m, glm.vec3(*position_mod))
        m = glm.rotate(m, math.radians(rz), glm.vec3(0.0, 0.0, 1.0))
        m = glm.rotate(m, math.radians(ry_mod), glm.vec3(0.0, 1.0, 0.0))
        m = glm.rotate(m, math.radians(rx), glm.vec3(1.0, 0.0, 0.0))
        return m


class Shader:
    """
    最低限のシェーダ実装
    ※OpenGLのコンテキストが有効になっている状態でインスタンス生成すること
    """

    def __init__(self) -> None:
        vertex_code = """
#version 330
in vec3 VertexPosition;  // VBO-buffer[0](頂点座標)に対応
in vec3 VertexColor;  // VBO-buffer[1](頂点色)に対応
out vec3 Color;  // 頂点色
uniform mat4 ModelMatrix;  // モデル行列
uniform mat4 ViewMatrix;  // ビュー行列
uniform mat4 ProjectionMatrix;  //射影行列
void main(void)
{
    Color = VertexColor;
    gl_Position = ProjectionMatrix * ViewMatrix * ModelMatrix * vec4(VertexPosition, 1.0);
}
"""
        fragment_code = """
#version 330        
in vec3 Color;  // vertex shaderの出力Color
out vec4 FragColor;  // 画素の色
void main(void){
    FragColor = vec4(Color, 1.0);
}    
"""
        # シェーダソースコードのコンパイル
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex_shader, vertex_code)
        glCompileShader(vertex_shader)
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader, fragment_code)
        glCompileShader(fragment_shader)

        # プログラムオブジェクトの作成とリンク
        program = glCreateProgram()
        glAttachShader(program, vertex_shader)
        glDeleteShader(vertex_shader)
        glAttachShader(program, fragment_shader)
        glDeleteShader(fragment_shader)
        glLinkProgram(program)
        self.program = program

        # 変数Locationの取得
        glUseProgram(self.program)
        self.loc_model = glGetUniformLocation(self.program, "ModelMatrix")
        self.loc_view = glGetUniformLocation(self.program, "ViewMatrix")
        self.loc_projection = glGetUniformLocation(self.program, "ProjectionMatrix")
        glUseProgram(0)

    def enable(self) -> None:
        """
        シェーダを有効にする。描画する前に必ず有効にすること
        """
        glUseProgram(self.program)

    def disable(self) -> None:
        """
        シェーダを無効にする
        :return:
        """
        glUseProgram(0)

    def set_model_matrix(self, m: 'glm.mat4x4') -> None:
        """
        シェーダにモデル行列を送る
        :param m: モデル行列(PyGLMのmat4x4形式)
        """
        glUniformMatrix4fv(self.loc_model, 1, GL_FALSE, glm.value_ptr(m))

    def set_view_matrix(self, m: 'glm.mat4x4') -> None:
        """
        シェーダにビュー行列を送る
        :param m: ビュー行列(PyGLMのmat4x4形式)
        """
        glUniformMatrix4fv(self.loc_view, 1, GL_FALSE, glm.value_ptr(m))

    def set_projection_matrix(self, m: 'glm.mat4x4') -> None:
        """
        シェーダに射影行列を送る
        :param m: 射影行列(PyGLMのmat4x4形式)
        """
        glUniformMatrix4fv(self.loc_projection, 1, GL_FALSE, glm.value_ptr(m))


class AppBase:
    """
    OpenGLアプリ用の汎用クラス
    派生クラスを作成し、runを呼び出すとOpenGLの最低限の機能が使えるようになるので、
    render関数等を適宜オーバーライドして拡張して使う
    """

    def __init__(self, width: int = 320, height: int = 320, caption: str = "OpenGL Demo"):
        """
        OpenGL用ウィンドウを生成する
        :param width: ウィンドウ幅(pixel)
        :param height: ウィンドウ高さ(pixel)
        :param caption: ウィンドウのタイトル
        """
        self.running = True

        # GLFW初期化
        if not glfw.init():
            raise RuntimeError('Failed on GLFW initialization')

        # ウィンドウ生成
        self.window = glfw.create_window(width, height, caption, None, None)
        if self.window is None:
            glfw.terminate()
            raise RuntimeError('Failed to create GLFW window')

        # OpenGLコンテキスト生成
        glfw.make_context_current(self.window)

        # OpenGL3.3を使う
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        # OpenGL基本設定
        glEnable(GL_CULL_FACE)  # カリングはON
        glCullFace(GL_BACK)  # 裏面を消去
        glEnable(GL_DEPTH_TEST)  # デプステストは有効

        # コールバック設定
        def on_key_event(*args):
            self.key_event(*args)

        glfw.set_key_callback(self.window, on_key_event)

        # シェーダ
        self.shader = Shader()
        self.shader.enable()

    def run(self):
        # メインループ
        while not glfw.window_should_close(self.window) and self.running:
            self.render()
            glfw.swap_buffers(self.window)
            glfw.poll_events()
        glfw.terminate()

    def render(self) -> None:
        """
        描画用関数。オーバーライドすること
        """
        pass

    def key_event(self, window, key, scancode, action, mods) -> None:
        """
        キー入力イベントの処理関数。必要に応じてオーバーライドすること
        引数については以下参照
        https://www.glfw.org/docs/latest/input_guide.html#input_key
        """
        # キー入力処理
        if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
            # アプリケーション終了
            self.running = False


class App(AppBase):
    """
    歯車回転デモアプリ(兼gear.make_figure_trianglesのサンプル)
    生成してrunを呼び出すと動く
    """
    def __init__(self):
        # ウィンドウの生成とOpenGL初期化
        WINDOW_WIDTH = 320
        WINDOW_HEIGHT = 320
        WINDOW_TITLE = "Gear 3D"
        super().__init__(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE)

        # 描画物の初期化
        common_params = {  # 共通パラメータ
            "height": 1.0,
            "m": 1.0,
            "alpha_deg": 20.0,
            "bottom_type": "line",
        }
        gear1_params = {**common_params, "z": 19, "r": 0.0, "g": 1.0, "b": 0.0}
        gear2_params = {**common_params, "z": 11, "r": 1.0, "g": 0.0, "b": 0.0}
        gear3_params = {**common_params, "z": 17, "r": 1.0, "g": 1.0, "b": 0.0}
        gear1 = GearModel(gear1_params)  # 歯車(駆動歯車)を作る
        gear1.set_angle(0.0)  # 速度を決める
        gear1.set_velocity(1.0)  # 初期角を決める
        gear2 = GearModel(gear2_params, gear1)  # 2は1に依存
        gear2.adjust_center(0.0)  # 依存対象を基準とした方位で位置を決定
        gear2.set_angle(18.0)  # かみ合って見えるように角度を調整
        gear3 = GearModel(gear3_params, gear1)  # 3は1に依存
        gear3.adjust_center(135.0)
        gear3.set_angle(1.0)
        self.gears = [gear1, gear2, gear3]
        self.camera_roll = 0.0  # デモでよくあるカメラをグルグル回す用

    def render(self):
        """
        描画処理を行う。
        """
        glClearColor(0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # ビュー行列を生成
        eye = [0.0, 30.0, 0.0]  # 視点
        target = [0.0, 0.0, 0.0]  # 注視点
        up = [0.0, 0.0, -1.0]  # 画面上方向
        view_matrix = glm.lookAt(glm.vec3(*eye), glm.vec3(*target), glm.vec3(*up))
        view_matrix = glm.rotate(view_matrix,  # 注視点を中心としてカメラをグルグル回す
                                 self.camera_roll, glm.vec3(0.0, 0.0, 1.0))
        self.camera_roll += math.radians(0.10)

        # 透視射影行列を生成
        view_angle = 90.0  # 視野角(degree)
        aspect_rate = 1.0
        z_near = 0.1  # これ以上近い物体は見えない
        z_far = 100.0  # これ以上遠い物体は見えない
        projection_matrix = glm.perspective(math.radians(view_angle),
                                            aspect_rate, z_near, z_far)

        # シェーダの設定を行う
        # シェーダ(self.shader)に対しては以下の3種類の行列を適切に供給する必要がある。
        # - モデル(model)行列: モデル毎に固有の行列。モデルの姿勢と位置を示す4x4正方行列
        # - ビュー(view)行列: カメラの姿勢と位置を示す4x4正方行列
        # - 射影(projection)行列: 射影方式とパラメータを表す4x4正方行列
        self.shader.set_view_matrix(view_matrix)  # ビュー行列を更新
        self.shader.set_projection_matrix(projection_matrix)  # 射影行列を更新

        # 歯車を描画
        for gear in self.gears:
            gear.update()  # 歯車の状態を更新
            model_matrix = gear.get_model_matrix()
            self.shader.set_model_matrix(model_matrix)  # モデル行列を更新
            gear.draw()  # 描画


def main():
    app = App()
    app.run()


if __name__ == "__main__":
    main()
