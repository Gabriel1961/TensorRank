import { Component } from "solid-js";
import styles from "./Home.module.scss"
import Typewriter from "./components/TypeWriter";
const HomePage: Component = () => {
  return <div class={styles.container}>
    <div class={styles.videoContainer}>
      <video autoplay muted loop class={styles.videoBackground}>
        <source src="/src/assets/matrixVideo.mp4" type="video/mp4"/>
      </video>
    </div>
    <div class={styles.header}>TensorRank</div>
    <Typewriter/>
  </div>
}

export default HomePage;