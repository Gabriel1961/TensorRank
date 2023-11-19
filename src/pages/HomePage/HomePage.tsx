import { Component } from "solid-js";
import styles from "./Home.module.scss"
import Typewriter from "./components/TypeWriter";
const HomePage: Component = () => {
  return <div class={styles.container}>
    <div class={styles.videoContainer}>
      <video autoplay muted loop class={styles.videoBackground}>
        <source src="https://firebasestorage.googleapis.com/v0/b/tensorrank-7126b.appspot.com/o/vecteezy_green-matrix-particle-fall-background-loop-animation_3663146.mp4?alt=media&token=ca80cf76-056c-4877-9adf-9d2fac8a9290" type="video/mp4"/>
      </video>
    </div>
    <div class={styles.header}>TensorRank</div>
    <Typewriter/>
  </div>
}

export default HomePage;