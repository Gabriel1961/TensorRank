import { Accessor, Component, createEffect, createSignal } from "solid-js";
import { Image } from "@kobalte/core";
import user from "../assets/user.svg"
import styles from "./ProfileImage.module.scss"
interface ProfileImageProps {
  getImageUrl?: Accessor<string | undefined>,
  onClick: () => void
}

const ProfileImage: Component<ProfileImageProps> = (props) => {
  return <Image.Root fallbackDelay={600} class={styles.image}>
    <Image.Img
      draggable={false}
      class={styles.image__img}
      src={props.getImageUrl ? props.getImageUrl() : undefined}
    />
    <Image.Fallback class={styles.image__fallback}>
      <img draggable={false} src={user} onClick={props.onClick} />
    </Image.Fallback>
  </Image.Root>
}

export default ProfileImage;