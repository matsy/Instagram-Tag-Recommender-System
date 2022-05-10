import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { MatGridListModule } from '@angular/material/grid-list';
import { Observable } from 'rxjs';

@Component({
  selector: 'app-hashtag-list',
  templateUrl: './hashtag-list.component.html',
  styleUrls: ['./hashtag-list.component.css']
})
export class HashtagListComponent implements OnInit {

  constructor(private http: HttpClient) { }

  private Url="http://35.202.9.1:5000/api/upload/captionimage";

  ngOnInit(): void {
  }
  
}
