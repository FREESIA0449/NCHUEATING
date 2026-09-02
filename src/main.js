import * as THREE from 'three/webgpu';
import { FontLoader, RoomEnvironment, TextGeometry } from 'three/examples/jsm/Addons.js';
import {
  instancedBufferAttribute,
  float,
  color,
  uv,
  metalness,
  hue,
  length,
  step,
  mix,
  vec3,
  positionLocal,
  roughness,
  storage,
  attribute,
  Fn,
  instanceIndex,
  mx_noise_vec3,
  time,
  uniform,
  rotate,
  mx_noise_float,
  transmission,
  dispersion,
  ior,
  instancedArray,
  thickness,
  sheen,
  iridescence,
  smoothstep,
  blur,
  oneMinus,
  sub,
  pass,
  mrt,
  output,
  normalView,
} from 'three/tsl';
import { bloom } from 'three/examples/jsm/tsl/display/BloomNode.js';
import { ao } from 'three/examples/jsm/tsl/display/GTAONode.js';
import { denoise } from 'three/examples/jsm/tsl/display/DenoiseNode.js';

const Resources = {
  font: undefined,
};

function preload() {
  // 安全兜底：字体加载挂起时，8s 后强制移除 loading，避免页面永久隐藏
  setTimeout(() => document.body.classList.remove('loading'), 8000);
  const _font_loader = new FontLoader();
  _font_loader.load(
    './assets/Times New Roman_Regular.json',
    (font) => {
      Resources.font = font;
      init();
    },
    undefined, // onProgress
    (err) => {
      console.error('Font loading failed:', err);
      document.body.classList.remove('loading');
      document.querySelector('.canvas-hint').style.display = 'none';
      document.body.classList.add('no-webgpu');
    }
  );
}

function init() {
  document.body.classList.remove('loading');

  // ── Scene setup ──
  const sizes = {
    width: window.innerWidth,
    height: window.innerHeight,
  };

  // WebGPU support check
  if (!navigator.gpu) {
    console.warn('WebGPU not supported in this browser. Falling back to static background.');
    const canvas = document.getElementById('canvas');
    if (canvas) canvas.style.display = 'none';
    document.querySelector('.canvas-hint').style.display = 'none';
    // 渐变只落在 hero 深色区（CSS: body.no-webgpu .hero），亮色内容区保持白色
    document.body.classList.add('no-webgpu');
    return;
  }

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, sizes.width / sizes.height, 0.1, 100);
  const renderer = new THREE.WebGPURenderer({ antialias: true, canvas: document.getElementById('canvas') });
  renderer.toneMapping = THREE.CineonToneMapping;
  renderer.toneMappingExposure = 0.5;
  // 触屏设备（手机）像素比上限 1.5，省电且肉眼几乎无差别
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, navigator.maxTouchPoints ? 1.5 : 2));
  document.body.appendChild(renderer.domElement);

  renderer.setSize(sizes.width, sizes.height);
  camera.position.z = 4.5;

  scene.add(camera);

  // ── Environment ──
  scene.fog = new THREE.Fog(new THREE.Color('#2a2e38'), 0.0, 10.0);
  scene.background = new THREE.Color('#1a1e27');

  const environment = new RoomEnvironment();
  const pmremGenerator = new THREE.PMREMGenerator(renderer);
  scene.environment = pmremGenerator.fromSceneAsync(environment).texture;
  scene.environmentIntensity = 0.8;

  const light = new THREE.DirectionalLight('#ffffff', 8);
  light.position.set(2.0, 2.0, 5.0);
  scene.add(light);

  const light2 = new THREE.AmbientLight('#8899bb', 1.5);
  scene.add(light2);

  // ── 3D Text: "EAT?" ──
  const text_geo = new TextGeometry('EAT?', {
    font: Resources.font,
    size: 1.3,
    depth: 0.25,
    bevelEnabled: true,
    bevelThickness: 0.1,
    bevelSize: 0.01,
    bevelOffset: 0,
    bevelSegments: 1,
  });

  // Center the text
  text_geo.computeBoundingBox();
  const centerOffset = -0.5 * (text_geo.boundingBox.max.x - text_geo.boundingBox.min.x);
  const centerOffsety = -0.5 * (text_geo.boundingBox.max.y - text_geo.boundingBox.min.y);
  text_geo.translate(centerOffset, centerOffsety - 0.55, 0);

  const mesh = new THREE.Mesh(
    text_geo,
    new THREE.MeshStandardMaterial({
      color: '#d0c8b8',
      metalness: 0.6,
      roughness: 0.25,
    })
  );

  scene.add(mesh);

  // ── Pointer interaction ──
  const u_input_pos = uniform(new THREE.Vector3(0, 0, 0));
  const u_input_pos_press = uniform(0.0);

  const ray_cast = new THREE.Raycaster();

  window.addEventListener('pointerup', () => {
    u_input_pos_press.value = 0.0;
  }, { passive: false });

  window.addEventListener('pointermove', (event) => {
    // 用户偏好减弱动效时，禁用指针爆炸形变（保留静置渲染）
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const x = event.clientX / sizes.width - 0.5;
    const y = event.clientY / sizes.height - 0.5;

    const _p = new THREE.Vector2(x, -y).multiplyScalar(2.0);
    ray_cast.setFromCamera(_p, camera);
    const intersect = ray_cast.intersectObject(mesh, true);
    if (intersect.length) {
      u_input_pos_press.value = 1.0;
      u_input_pos.value.copy(intersect[0].point);
    }
  }, { passive: false });

  // ── Uniforms ──
  const u_noise_amp = uniform(1.6);
  const u_spring = uniform(0.05);
  const u_friction = uniform(0.9);
  const u_explode_amp = uniform(1.5);

  const count = text_geo.attributes.position.count;
  const initial_position = storage(text_geo.attributes.position, 'vec3', count);
  const normal_at = storage(text_geo.attributes.normal, 'vec3', count);

  const position_storage_at = storage(new THREE.StorageBufferAttribute(count, 3), 'vec3', count);
  const velocity_storage_at = storage(new THREE.StorageBufferAttribute(count, 3), 'vec3', count);

  const compute_init = Fn(() => {
    position_storage_at.element(instanceIndex).assign(initial_position.element(instanceIndex));
    velocity_storage_at.element(instanceIndex).assign(vec3(0.0, 0.0, 0.0));
  })().compute(count);

  renderer.computeAsync(compute_init);

  const compute_update = Fn(() => {
    const base_position = initial_position.element(instanceIndex);
    const current_position = position_storage_at.element(instanceIndex);
    const current_velocity = velocity_storage_at.element(instanceIndex);
    const normal = normal_at.element(instanceIndex);

    const noise = mx_noise_vec3(current_position.mul(0.5).add(vec3(0.0, time, 0.0)), 1.0).mul(u_noise_amp);
    const distance = length(u_input_pos.sub(base_position));
    const pointer_influence = step(distance, 0.5).mul(u_explode_amp);

    const disorted_pos = base_position.add(noise.mul(normal.mul(pointer_influence)));
    disorted_pos.assign(rotate(disorted_pos, vec3(normal.mul(distance)).mul(pointer_influence)));
    disorted_pos.assign(mix(base_position, disorted_pos, u_input_pos_press));

    current_velocity.addAssign(disorted_pos.sub(current_position).mul(u_spring));
    current_position.addAssign(current_velocity);
    current_velocity.assign(current_velocity.mul(u_friction));
  })().compute(count);

  mesh.material.positionNode = position_storage_at.toAttribute();

  // ── Emissive color based on velocity ──
  const emissive_color = color('#ff6a2c');
  const emissive_bust = uniform(4.0);
  const vel_at = velocity_storage_at.toAttribute();
  const hue_rotated = vel_at.mul(Math.PI * 10.0);
  const emission_factor = length(vel_at).mul(10.0);
  mesh.material.emissiveNode = hue(emissive_color, hue_rotated).mul(emission_factor).mul(emissive_bust);

  // ── Post Processing ──
  const composer = new THREE.PostProcessing(renderer);
  const scene_pass = pass(scene, camera);

  scene_pass.setMRT(
    mrt({
      output: output,
      normal: normalView,
    })
  );

  const scene_color = scene_pass.getTextureNode('output');
  const scene_depth = scene_pass.getTextureNode('depth');
  const scene_normal = scene_pass.getTextureNode('normal');

  const ao_pass = ao(scene_depth, scene_normal, camera);
  ao_pass.resolutionScale = 1.0;

  const ao_denoise = denoise(ao_pass.getTextureNode(), scene_depth, scene_normal, camera).mul(scene_color);
  const bloom_pass = bloom(ao_denoise, 0.5, 0.2, 0.1);
  const post_noise = mx_noise_float(vec3(uv(), time.mul(0.1)).mul(sizes.width), 0.03).mul(1.0);

  composer.outputNode = ao_denoise.add(bloom_pass).add(post_noise);

  // ── Render loop（hero 视图隐藏或页面切后台时暂停，省电省 GPU） ──
  function animate() {
    renderer.computeAsync(compute_update);
    composer.renderAsync();
  }
  let running = true;
  const heroEl = document.querySelector('.hero');
  const heroObs = new IntersectionObserver((entries) => {
    running = entries[0].isIntersecting && !document.hidden;
    renderer.setAnimationLoop(running ? animate : null);
  });
  heroObs.observe(heroEl);
  document.addEventListener('visibilitychange', () => {
    const heroVisible = heroEl.getBoundingClientRect().height > 0
      && getComputedStyle(heroEl).display !== 'none';
    running = heroVisible && !document.hidden;
    renderer.setAnimationLoop(running ? animate : null);
  });
  renderer.setAnimationLoop(animate);

  // ── Resize（visualViewport：移动端地址栏收展不拉伸 canvas） ──
  const vv = window.visualViewport;
  function onResize() {
    sizes.width = vv ? vv.width : window.innerWidth;
    sizes.height = vv ? vv.height : window.innerHeight;
    camera.aspect = sizes.width / sizes.height;
    camera.updateProjectionMatrix();
    renderer.setSize(sizes.width, sizes.height);
  }
  (vv || window).addEventListener('resize', onResize);
  window.addEventListener('orientationchange', onResize);
}

window.onload = preload;
